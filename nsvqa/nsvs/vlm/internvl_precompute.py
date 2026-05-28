from transformers import AutoModel, AutoTokenizer
from torch.nn.functional import softmax
import numpy as np
import logging
import copy
import torch
import gc
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu
import torchvision.transforms as T
from PIL import Image
import math
from transformers.cache_utils import DynamicCache

from nsvqa.utils.sigmoid import calibrate_sigmoid
from nsvqa.nsvs.vlm.obj import DetectedObject
import transformers
import traceback

logging.basicConfig(level=logging.INFO)


class InternVL:
    """InternVL's Vision Language Model with KV-cache reuse across questions."""

    def __init__(
        self,
        model_name: str = "InternVL2-8B",
        multi_gpus: bool = False,
        device: int = 0,
        max_patch: int = 6,
    ) -> None:
        """Initialization the InternVL."""
        logging.info(
            (
                "You are using the model based on HuggingFace API.",
                "The model will be downloaded to the HuggingFace cache dir.",
            )
        )
        self.model_name = model_name
        self._path = f"OpenGVLab/{model_name}"
        self._num_gpus = torch.cuda.device_count()
        self.device = device
        if multi_gpus:
            device_map = split_model(model_name)
        else:
            device_map = assign_device_map(model_name=model_name, manual_gpu_id=device)
        self.model = AutoModel.from_pretrained(
            self._path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            device_map=device_map,
        )
        self.model.apply(self.move_tensors_to_gpu)
        self.model.eval()
        self.max_patch = max_patch
        self.model.config.max_dynamic_patch = self.max_patch
        logging.info(f"Using dynamic batch {self.model.config.max_dynamic_patch}")
        self.model.apply(self.move_tensors_to_gpu)
        self.tokenizer = AutoTokenizer.from_pretrained(self._path, trust_remote_code=True, use_fast=False)
        # InternLM2 tokenizer sometimes has pad_token_id=None; make sure it's set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.num_frame = 32
        self.IMG_START_TOKEN = '<img>'
        self.IMG_END_TOKEN = '</img>'
        self.IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

    def reset_model(self) -> None:
        """Reset the model to its initial state using pretrained weights."""
        self.model = AutoModel.from_pretrained(
            self._path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        ).eval()
        self.model.apply(self.move_tensors_to_gpu)

    def clear_gpu_memory(self) -> None:
        """Clear CUDA cache and run garbage collection to free GPU memory."""
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
        gc.collect()

    def move_tensors_to_gpu(
        self,
        module: torch.nn.Module,
    ) -> None:
        """Move all tensors in the module to GPU if they are on the CPU."""
        for name, tensor in module.named_buffers():
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu":
                module.register_buffer(
                    name,
                    tensor.cuda(self.device),
                    persistent=False,
                )
        for _, param in module.named_parameters():
            if param.device.type == "cpu":
                param.data = param.data.cuda(self.device)

    def batch_detect(
        self,
        seq_of_frames: list[np.ndarray],
        scene_descriptions: list[str],
        threshold: float
    ) -> [DetectedObject]:  # pyright: ignore[reportInvalidTypeForm]
        """Detect objects in the given frame image.

        Args:
            seq_of_frames (list[np.ndarray]): List of video frames to process.
            scene_descriptions (list[str]): Description of the scene.
            threshold (float): Detection threshold.

        Returns:
            DetectedObject: Detected objects with their details.
        """
        task_items = []
        for scene_description in scene_descriptions:
            if "subtitle" in scene_description:
                subtitle_scene_description = scene_description.replace("subtitle_", "").replace("_", " ")
                parsing_rule = "You must only return a Yes or No, and not both, to any question asked. You must not include any other symbols, information, text, justification in your answer or repeat Yes or No multiple times. For example, if the question is \"Does the video have the subtitle 'this is very interesting' present in the sequence of images?\", the answer must only be 'Yes' or 'No'."
                prompt = rf"Does the video have the subtitle '{subtitle_scene_description}' present in the sequence of images? " f"\n[PARSING RULE]: {parsing_rule}"
            else:
                if scene_description.startswith("n_"):
                    scene_description = scene_description[2:]
                object_scene_description = scene_description.replace("_", " ")
                parsing_rule = "You must only return a Yes or No, and not both, to any question asked. You must not include any other symbols, information, text, justification in your answer or repeat Yes or No multiple times. For example, if the question is \"Is there a cat present in the sequence of images?\", the answer must only be 'Yes' or 'No'."
                prompt = rf"Is there a '{object_scene_description}' present in the sequence of images? " f"\n[PARSING RULE]: {parsing_rule}"
            task_items.append({
                "name": scene_description,
                "prompt": prompt
            })

        prompts = [item["prompt"] for item in task_items]

        try:
            responses, confidences = self.infer_with_video_confidence_batch(
                languages=prompts,
                seq_of_frames=seq_of_frames
            )
        except Exception as e:
            logging.info(f"CRITICAL ERROR: {e}")
            raise e

        detected_objects = []
        for item, response, confidence in zip(task_items, responses, confidences):
            detected = "yes" in response.lower()

            probability = calibrate_sigmoid(confidence, false_threshold=threshold)

            obj = DetectedObject(
                name=item["name"],
                is_detected=detected,
                confidence=round(confidence, 3),
                probability=round(probability, 3),
            )
            detected_objects.append(obj)

        return detected_objects

    # ------------------------------------------------------------------
    # KV-cache reuse path: prefill visual prefix once, answer N questions
    # ------------------------------------------------------------------

    def _build_visual_prefix_text(self, num_patches_list):
        """Build the prompt prefix: system + user role open + all frame image tokens.

        Stops BEFORE the question, so this text is question-independent.
        Returns (prefix_text, template). The template is returned so the per-question
        suffix can be assembled consistently (sep, assistant role, etc).
        """
        template = copy.deepcopy(self.model.conv_template)
        template.system_message = self.model.system_message

        # Build prefix manually so it is exactly question-independent.
        # For InternLM2 chat this is: "<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n"
        prefix = template.system_template.format(system_message=template.system_message)
        prefix += template.sep + template.roles[0] + "\n"

        # Inject frame image tokens — same format as the original batch_chat.
        video_prefix = ""
        for i, num_patches in enumerate(num_patches_list):
            context_tokens = (
                self.IMG_CONTEXT_TOKEN
                * self.model.num_image_token
                * num_patches
            )
            video_prefix += (
                f"Frame{i + 1}: "
                + self.IMG_START_TOKEN + context_tokens + self.IMG_END_TOKEN
                + "\n"
            )
        prefix += video_prefix
        return prefix, template

    @torch.no_grad()
    def prefill_visual_prefix(self, seq_of_frames):
        """Run ViT + LM prefill over the visual prefix. Returns cache + metadata.

        This is the expensive, question-independent work. Call once per window of
        frames, then call `answer_with_cache` repeatedly for each question.
        """
        # 1. Vision side
        pixel_values, num_patches_list = load_video_from_seq_of_frames(
            seq_of_frames=seq_of_frames, device=self.device, max_num=self.max_patch
        )

        img_context_token_id = self.tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        vit_embeds = self.model.extract_feature(pixel_values)        # [P, T_img, H]
        vit_embeds = vit_embeds.reshape(-1, vit_embeds.shape[-1])    # [P*T_img, H]

        # 2. Build prefix text and tokenize
        prefix_text, template = self._build_visual_prefix_text(num_patches_list)
        enc = self.tokenizer(prefix_text, return_tensors='pt', add_special_tokens=False)
        input_ids = enc.input_ids.to(self.device)
        attention_mask = enc.attention_mask.to(self.device)

        # 3. Embed and splice vision features into IMG_CONTEXT slots
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()
        selected = (input_ids == img_context_token_id)
        assert selected.sum() == vit_embeds.shape[0], (
            f"IMG_CONTEXT count {selected.sum().item()} != vision feature count {vit_embeds.shape[0]}"
        )
        input_embeds[selected] = vit_embeds.to(input_embeds.dtype)

        # 4. Prefill the LM — pass None so the model creates the cache internally
        output = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values

        # InternLM2 only accepts tuple-of-tuples cache, not DynamicCache
        if isinstance(cache, DynamicCache):
            cache = tuple(
                (k, v) for k, v in zip(cache.key_cache, cache.value_cache)
            )

        return {
            'cache': cache,
            'attention_mask': attention_mask,   # shape [1, prefix_len]
            'prefix_len': input_ids.shape[1],
            'template': template,
        }

    @staticmethod
    def _clone_cache(cache):
        """Deep-clone a tuple-of-tuples KV cache so per-question generation
        does not contaminate the shared prefix cache."""
        return tuple((k.clone(), v.clone()) for k, v in cache)

    @torch.no_grad()
    def answer_with_cache(self, prefix, question, max_new_tokens=1):
        """Answer a yes/no question reusing the prefilled visual prefix KV cache.

        Uses a single forward pass instead of generate() — generate()'s
        prepare_inputs_for_generation assumes input_ids starts from position 0,
        which conflicts with our pre-filled cache.

        Args:
            prefix: dict returned by `prefill_visual_prefix`.
            question: question string (no image placeholders).
            max_new_tokens: ignored (kept for API compat), always generates 1 token.

        Returns:
            (response_text, confidence_float)
        """
        template = prefix['template']
        prefix_len = prefix['prefix_len']

        suffix = question + template.sep + template.roles[1] + "\n"
        q_ids = self.tokenizer(
            suffix, return_tensors='pt', add_special_tokens=False
        ).input_ids.to(self.device)
        q_len = q_ids.shape[1]

        cache = self._clone_cache(prefix['cache'])

        # Attention mask covers prefix (in cache) + question tokens
        full_attn = torch.cat(
            [prefix['attention_mask'], torch.ones_like(q_ids)], dim=1
        )

        # Position IDs for question tokens continue from where prefix ended
        position_ids = torch.arange(
            prefix_len, prefix_len + q_len, device=self.device
        ).unsqueeze(0)

        # Single forward pass — no generate() loop needed for yes/no
        output = self.model.language_model(
            input_ids=q_ids,
            attention_mask=full_attn,
            past_key_values=cache,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )

        # Logits at the last position = prediction for the next token
        last_logits = output.logits[:, -1, :]
        next_token_id = last_logits.argmax(dim=-1).item()

        response = self.tokenizer.decode(next_token_id, skip_special_tokens=True).strip()

        # Confidence: softmax probability of the greedy token
        probs = torch.softmax(last_logits, dim=-1)
        conf = probs[0, next_token_id].item()

        return response, conf

    def infer_with_video_confidence_batch(
        self,
        languages: list[str],
        seq_of_frames: list[np.ndarray],
        max_new_tokens: int = 128,
        do_sample: bool = False,
    ) -> tuple[list[str], list[float]]:
        """Run VQA for many questions against the same window of frames.

        Prefills the visual prefix once and reuses its KV cache for every
        question. `do_sample` is accepted for API compatibility but ignored:
        confidence calculation requires greedy decoding.

        Args:
            languages: list of question strings.
            seq_of_frames: list of frames (numpy arrays) for this window.
            max_new_tokens: cap on generated answer length per question.
            do_sample: ignored — generation is always greedy here.

        Returns:
            (responses, confidences) — parallel lists.
        """
        prefix = self.prefill_visual_prefix(seq_of_frames)
        responses, confidences = [], []
        for q in languages:
            r, c = self.answer_with_cache(prefix, q, max_new_tokens=max_new_tokens)
            responses.append(r)
            confidences.append(c)
        # Free the cache before returning so the next call starts clean
        del prefix
        return responses, confidences

    # ------------------------------------------------------------------
    # Legacy path: kept for reference / fallback. Not used by batch_detect.
    # ------------------------------------------------------------------

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        """Legacy batched chat that re-prefills the visual prefix for every
        question. Superseded by `infer_with_video_confidence_batch`, which uses
        KV-cache reuse. Kept here in case callers still depend on it."""

        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        total_patches_idx = 0
        for question in questions:
            template = copy.deepcopy(self.model.conv_template)
            template.system_message = self.model.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            num_images_this_question = query.count('<image>')
            for _ in range(num_images_this_question):
                num_patches = num_patches_list[total_patches_idx]
                context_tokens = IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches
                image_tokens = IMG_START_TOKEN + context_tokens + IMG_END_TOKEN
                query = query.replace("<image>", image_tokens, 1)
                total_patches_idx += 1
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True, add_special_tokens=False)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)

        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['do_sample'] = False
        generation_config['eos_token_id'] = eos_token_id
        generation_config["return_dict_in_generate"] = True
        generation_config["min_new_tokens"] = 1
        generation_config["output_scores"] = True
        generation_config["pad_token_id"] = tokenizer.pad_token_id

        hidden_dim = self.model.language_model.config.hidden_size

        with torch.no_grad():
            single_window_features = self.model.extract_feature(pixel_values)

        visual_block = single_window_features.view(1, -1, hidden_dim)
        batch_size = len(questions)
        expanded_features = visual_block.expand(batch_size, -1, -1).contiguous()

        try:
            generation_output = self.model.generate(
                pixel_values=pixel_values[0:1],
                visual_features=expanded_features,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config
            )
        except Exception as e:
            print(f"ERROR during generate: {e}")
            traceback.print_exc()
            raise

        gen_sequences = generation_output.sequences

        responses = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip().rstrip('.').strip() for response in responses]

        batch_confidences = []
        for b in range(gen_sequences.shape[0]):
            item_tokens = gen_sequences[b]
            token_probs = []

            for step_idx in range(len(item_tokens)):
                if step_idx >= len(generation_output.scores):
                    break

                token_id = item_tokens[step_idx].item()

                if token_id == tokenizer.eos_token_id:
                    break

                logits = generation_output.scores[step_idx][b]
                probs = torch.softmax(logits, dim=-1)
                token_probs.append(probs[token_id].item())

            if token_probs:
                conf = np.exp(np.mean(np.log(np.array(token_probs) + 1e-9)))
                batch_confidences.append(float(conf))
            else:
                batch_confidences.append(0.0)
        return responses, batch_confidences

    def vqa(self, question, video, generation_config=None):
        """
        Video Question Answering inference for a single question.

        Args:
            question (str): The text question.
            video (str or dict): Either a string file path or a dictionary containing
                                 {'path': str, 'segments': list} for RT-NeuS logic.
            generation_config (dict, optional): Configuration for text generation.
        """
        if generation_config is None:
            generation_config = dict(max_new_tokens=1024, do_sample=False)

        # --- 1. Load Video Pixel Values ---
        if isinstance(video, dict):
            if "segments" in video:
                video_path = video["path"]
                segments = video["segments"]
                pixel_values, num_patches_list = load_video_uniformly_from_segments(
                    video_path,
                    segments=segments,
                    num_segments=self.num_frame
                )
            else:
                video_path = video.get("path")
                print(f'[DEBUG] Reading Full Video: {video_path}')
                pixel_values, num_patches_list = load_video(
                    video_path,
                    num_segments=self.num_frame
                )
        else:
            pixel_values, num_patches_list = load_video(
                video,
                num_segments=self.num_frame
            )

        # --- 2. Prepare Tensors ---
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        # --- 3. Construct Prompt ---
        video_prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])
        full_prompt = video_prefix + question

        # --- 4. Generate Response ---
        response, history = self.model.chat(
            self.tokenizer,
            pixel_values,
            full_prompt,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=True
        )

        return response


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int) -> T.Compose:
    """Builds a transformation pipeline for the given input size."""
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def load_video_uniformly_from_segments(video_path, segments, input_size=448, max_num=1, num_segments=32):
    """
    Simulates uniform sampling from a cropped video by mapping
    global indices back to the original non-contiguous frame indices.
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    transform = build_transform(input_size=input_size)

    # 1. Map segments to a single "Virtual Timeline"
    virtual_timeline = []
    for start, end in segments:
        virtual_timeline.extend(range(int(start), int(end) + 1))

    total_virtual_frames = len(virtual_timeline)

    # 2. Use the original logic to get indices across the virtual total length
    seg_size = float(total_virtual_frames) / num_segments

    # These are indices into our 'virtual_timeline' list
    virtual_indices = [int((seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)]

    # Clip to avoid index errors at the very end
    virtual_indices = [min(i, total_virtual_frames - 1) for i in virtual_indices]

    # 3. Map virtual indices back to original video frame indices
    original_frame_indices = [virtual_timeline[i] for i in virtual_indices]

    # 4. Extract and Process
    pixel_values_list, num_patches_list = [], []
    frames = vr.get_batch(original_frame_indices).asnumpy()

    for frame in frames:
        img = Image.fromarray(frame).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def assign_device_map(model_name, manual_gpu_id=0):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        "InternVL2-1B": 24,
        "InternVL2-2B": 24,
        "InternVL2-4B": 32,
        "InternVL2-8B": 32,
        "InternVL2-26B": 48,
        "InternVL2-40B": 60,
        "InternVL2-Llama3-76B": 80,
    }[model_name]
    print(f"Device is {manual_gpu_id}, model will be loaded there")
    for layer_idx in range(num_layers):
        device_map[f"language_model.model.layers.{layer_idx}"] = manual_gpu_id

    device_map["vision_model"] = manual_gpu_id
    device_map["mlp1"] = manual_gpu_id
    device_map["language_model.model.tok_embeddings"] = manual_gpu_id
    device_map["language_model.model.embed_tokens"] = manual_gpu_id
    device_map["language_model.output"] = manual_gpu_id
    device_map["language_model.model.norm"] = manual_gpu_id
    device_map["language_model.lm_head"] = manual_gpu_id
    device_map[f"language_model.model.layers.{num_layers - 1}"] = manual_gpu_id

    print(device_map)
    return device_map


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        "InternVL2-1B": 24,
        "InternVL2-2B": 24,
        "InternVL2-4B": 32,
        "InternVL2-8B": 32,
        "InternVL2-26B": 48,
        "InternVL2-40B": 60,
        "InternVL2-Llama3-76B": 80,
    }[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map


def move_tensors_to_gpu(module):
    for name, tensor in module.named_buffers():
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu":
            module.register_buffer(name, tensor.cuda(), persistent=False)
    for _, param in module.named_parameters():
        if param.device.type == "cpu":
            param.data = param.data.cuda()


# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)]
    )
    return frame_indices


def load_video_from_seq_of_frames(
    seq_of_frames: list[np.ndarray],
    input_size=448,
    max_num=1,
    device="cuda",
    dtype=torch.bfloat16,
):
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    for img in seq_of_frames:
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values).to(dtype=dtype, device=device)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    return torch.cat(pixel_values_list, dim=0), num_patches_list


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values.to(torch.bfloat16))
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list