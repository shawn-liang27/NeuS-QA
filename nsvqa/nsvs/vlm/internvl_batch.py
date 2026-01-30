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

from nsvqa.utils.sigmoid import calibrate_sigmoid 
from nsvqa.nsvs.vlm.obj import DetectedObject
import transformers
import traceback
from torch.backends.cuda import sdp_kernel

logging.basicConfig(level=logging.INFO)


# # Set the library-wide logging level to DEBUG
# transformers.utils.logging.set_verbosity_debug()

# # Optional: specifically target the internvl-chat components if you have the source
# logging.getLogger("transformers").setLevel(logging.DEBUG)

class InternVL:
    """InternVL's Vision Language Model."""

    def __init__(
        self,
        model_name: str = "InternVL2-8B",
        multi_gpus: bool = False,
        device: int = 0,
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
            use_flash_attn=False,
            attn_implementation="eager",
            trust_remote_code=True,
            device_map=device_map,
        )
        self.model.eval()
        self.model.config.max_dynamic_patch = 12
        logging.info(f"Using dynamic batch {self.model.config.max_dynamic_patch}")
        self.model.apply(self.move_tensors_to_gpu)
        self.tokenizer = AutoTokenizer.from_pretrained(self._path, trust_remote_code=True, use_fast=False)

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
        gc.collect()  # Run garbage collector

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
    ) -> [DetectedObject]:
        """Detect objects in the given frame image.

        Args:
            seq_of_frames (list[np.ndarray]): List of video frames to process.
            scene_descriptions (list[str]): Description of the scene.
            threshold (float): Detection threshold.

        Returns:
            DetectedObject: Detected objects with their details.
        """
        prompts = []
        for scene_description in scene_descriptions:
            if "subtitle" in scene_description:
                subtitle_scene_description = scene_description.replace("subtitle_", "").replace("_", " ")
                parsing_rule = "You must only return a Yes or No, and not both, to any question asked. You must not include any other symbols, information, text, justification in your answer or repeat Yes or No multiple times. For example, if the question is \"Does the video have the subtitle 'this is very interesting' present in the sequence of images?\", the answer must only be 'Yes' or 'No'."
                prompt = rf"Does the video have the subtitle '{subtitle_scene_description}' present in the sequence of images? " f"\n[PARSING RULE]: {parsing_rule}"
                prompts.append(prompt)
            else:
                object_scene_description = scene_description.replace("_", " ")
                parsing_rule = "You must only return a Yes or No, and not both, to any question asked. You must not include any other symbols, information, text, justification in your answer or repeat Yes or No multiple times. For example, if the question is \"Is there a cat present in the sequence of images?\", the answer must only be 'Yes' or 'No'."
                prompt = rf"Is there a '{object_scene_description}' present in the sequence of images? " f"\n[PARSING RULE]: {parsing_rule}"
                prompts.append(prompt)
        try:
            
            responses, confidences = self.infer_with_video_confidence_batch(
                languages=prompts,        # List of questions
                seq_of_frames=seq_of_frames # The batch of frames
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.info(f"CRITICAL: CUDA OOM - Batch size: {len(seq_of_frames)}")
            raise e

        detected_objects = []

        for response, confidence in zip(responses, confidences):
            detected = "yes" in response.lower()
            
            probability = calibrate_sigmoid(confidence, false_threshold=threshold)
            
            obj = DetectedObject(
                name=scene_description,
                is_detected=detected,
                confidence=round(confidence, 3),
                probability=round(probability, 3),
            )
            detected_objects.append(obj)

        return detected_objects

    def infer_with_video_confidence(
        self,
        languages: str,
        seq_of_frames: list[np.ndarray],
        max_new_tokens: int = 1024,
        do_sample: bool = True,
    ) -> tuple[str, float]:
        """Perform video inference and return response with confidence score.

        Args:
            language (str): The input prompt or question.
            seq_of_frames (list[np.ndarray] | None):
                List of video frames as numpy arrays.
            video_path (str | None): Path to the input video file.
            max_new_tokens (int): Maximum number of new tokens to generate.
            do_sample (bool): Whether to use sampling for generation.

        Returns:
            list[tuple[str, float]]: Generated response and confidence score.
        """

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }

        pixel_values, num_patches_list = load_video_from_seq_of_frames(
            seq_of_frames=seq_of_frames, device=self.device
        )

        video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
        language = video_prefix + language
        languages = [language = video_prefix + language for language in languages]
        return self.batch_chat(
            self.tokenizer,
            pixel_values,
            languages,
            generation_config,
            num_patches_list=num_patches_list,
            verbose=False
        )

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        device = torch.device(self.language_model.device if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_config["return_dict_in_generate"] = True
        generation_config["output_scores"] = True
        generation_config["output_logits"] = True
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )

        
        input_token_len = input_ids.shape[1]
        gen_sequences = generation_output.sequences[:, input_token_len:]

        responses = tokenizer.batch_decode(gen_sequences, skip_special_tokens=True)

        responses = [res.split(template.sep)[0].strip() for res in responses]

        batch_confidences = []
        eos_token_id = tokenizer.eos_token_id

        for b in range(gen_sequences.shape[0]):
            confidence = 1.0
            
            for i in range(gen_sequences.shape[1]):
                token_id = gen_sequences[b, i].item()
                
                # Stop calculating if we hit EOS
                if token_id == eos_token_id:
                    break
                    
                # generation_output.scores is a tuple of length (generated_tokens)
                # Each element is [batch_size, vocab_size]
                if i < len(generation_output.scores):
                    step_logits = generation_output.scores[i][b] # [vocab_size]
                    probs = torch.softmax(step_logits, dim=-1)
                    token_prob = probs[token_id].item()
                    confidence *= token_prob
                    
            batch_confidences.append(confidence)
        return responses, batch_confidences

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
        pixel_values = torch.stack(pixel_values).to(dtype=dtype, device=device)  # Convert to bfloat16
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