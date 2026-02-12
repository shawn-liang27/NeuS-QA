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
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


# Set the library-wide logging level to DEBUG
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
        max_patches: int = 6,
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
        self.max_patches = max_patches
        self.model.config.max_dynamic_patch = self.max_patches
        logging.info(f"Using dynamic batch {self.model.config.max_dynamic_patch}")
        self.model.apply(self.move_tensors_to_gpu)
        self.tokenizer = AutoTokenizer.from_pretrained(self._path, trust_remote_code=True)

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

    def load_video_from_seq_of_frames(
        self,
        seq_of_frames: list[np.ndarray],
        input_size=448,
        max_num=1,
        device="cuda",
        dtype=torch.bfloat16,
    ):
        pixel_values_list, num_patches_list = [], []
        transform = build_transform(input_size=input_size)
        for img in tqdm(seq_of_frames):
            img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        return torch.cat(pixel_values_list, dim=0), num_patches_list


    def prepare_batch_inputs(self, scene_descriptions, patches_per_window, IMG_CONTEXT_TOKEN="<IMG_CONTEXT>"):
        """
        Compute input_ids once for the entire batch of windows.
        Since questions are the same across all windows, we only need one set of input_ids.
        """
        # 1. Build the prompts
        task_items = []

        for desc in scene_descriptions:
            if "subtitle" in desc:
                clean_desc = desc.replace("subtitle_", "").replace("_", " ")
                prompt = (
                    f"Does the video have the subtitle '{clean_desc}' present in the sequence of images?\n"
                    f"[PARSING RULE]: Answer ONLY 'Yes' or 'No'. Do not include punctuation or explanations. "
                    f"Example: Yes"
                )
            else:
                clean_desc = desc.replace("_", " ")
                prompt = (
                    f"Does the video have '{clean_desc}' present in the sequence of images?\n"
                    f"[PARSING RULE]: Answer ONLY 'Yes' or 'No'. Do not include punctuation or explanations. "
                    f"Example: Yes"
                )
            task_items.append({
                "name": clean_desc,
                "prompt": prompt
            })

        clean_descs = [item["name"] for item in task_items]
        prompts = [item["prompt"] for item in task_items]
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        # 2. Build the visual prefix string
        # Frame1: <img><IMG_CONTEXT>...</img>\nFrame2: ...
        img_context = IMG_CONTEXT_TOKEN * self.model.num_image_token
        visual_prefix = ""
        for i, num_patches in enumerate(patches_per_window):
            visual_prefix += f"Frame{i+1}: <img>{img_context * num_patches}</img>\n"

        # 3. Final Tokenization (One pass for the whole set of questions)
        queries = []
        for p in prompts:
            template = copy.deepcopy(self.model.conv_template)
            template.system_message = self.model.system_message
            template.append_message(template.roles[0], visual_prefix + p)
            template.append_message(template.roles[1], None)
            queries.append(template.get_prompt())
        eos_token_id = self.tokenizer.convert_tokens_to_ids(self.model.conv_template.sep.strip())
        self.tokenizer.padding_side = 'left'
        tokens = self.tokenizer(queries, return_tensors='pt', padding=True, add_special_tokens=False)
        
        return {
            'input_ids': tokens['input_ids'].to(self.device),
            'attention_mask': tokens['attention_mask'].to(self.device),
            'eos_token_id' : eos_token_id,
            'propositions' : clean_descs
        }

    def batch_detect_multi_window(
        self,
        scene_descriptions,
        batch_pixels_cpu,
        precomputed_inputs,
        batch_size,
        threshold: float
    ) -> [DetectedObject]:
        try:
            responses, confidences = self.batch_chat(
                tokenizer=self.tokenizer,
                batch_pixels_cpu=batch_pixels_cpu,
                precomputed_inputs=precomputed_inputs,
                batch_size=batch_size
            )
        except Exception as e:
            logging.info(f"CRITICAL ERROR: {e}")
            raise e

        num_questions = len(scene_descriptions)
        all_window_results = []
        for i in range(batch_size):
            window_objects = []
            # Calculate the slice for this specific window's responses
            start_idx = i * num_questions
            end_idx = start_idx + num_questions
            
            window_responses = responses[start_idx:end_idx]
            window_confidences = confidences[start_idx:end_idx]

            for name, resp, conf in zip(scene_descriptions, window_responses, window_confidences):
                detected = "yes" in resp.lower()
                probability = calibrate_sigmoid(conf, false_threshold=threshold)
                
                obj = DetectedObject(
                    name=name,
                    is_detected=detected,
                    confidence=round(conf, 3),
                    probability=round(probability, 3),
                )
                window_objects.append(obj)
                
            all_window_results.append(window_objects)

        return all_window_results

    def batch_chat(self, tokenizer, batch_pixels_cpu, precomputed_inputs, batch_size):
        """
        The simplified high-speed inference loop.
        """
        # 1. Move only current window to GPU
        tokens_per_tile = (self.model.num_image_token)
        gpu_pixels = batch_pixels_cpu.to(self.device, dtype=torch.bfloat16)

        generation_config = {
            "max_new_tokens": 16,
        }
        generation_config['do_sample'] = False
        generation_config['eos_token_id'] = precomputed_inputs["eos_token_id"]
        generation_config["return_dict_in_generate"] = True
        generation_config["output_scores"] = True
        generation_config["pad_token_id"] = tokenizer.pad_token_id

        hidden_dim = self.model.language_model.config.hidden_size
        
        with torch.no_grad():
            # 2. Extract Features once
            all_features = self.model.extract_feature(gpu_pixels)
        
            tiles_per_window = (all_features.shape[0] // batch_size) * tokens_per_tile

            window_features = all_features.view(batch_size, tiles_per_window, -1)
            # Flatten the middle dimensions so it's a sequence of tokens
            # New Shape: [Batch_Size, tiles_per_window * 256, Hidden_Dim]
            window_features = window_features.reshape(batch_size, -1, all_features.shape[-1])

            num_questions = precomputed_inputs['input_ids'].shape[0]
        
            # Match each window to its set of questions
            expanded_features = window_features.repeat_interleave(num_questions, dim=0)
            expanded_input_ids = precomputed_inputs['input_ids'].repeat(batch_size, 1)
            expanded_attn_mask = precomputed_inputs['attention_mask'].repeat(batch_size, 1)

            try:
                # 4. Generate with shared features
                generation_output = self.model.generate(
                    pixel_values=gpu_pixels[0:1], # Reference pixel
                    visual_features=expanded_features,
                    input_ids=expanded_input_ids,
                    attention_mask=expanded_attn_mask,
                    **generation_config
                )
            except Exception as e:
                print(f"ERROR during generate: {e}")
                import traceback
                traceback.print_exc()

        gen_sequences = generation_output.sequences 

        responses = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
        responses = [response.split(self.model.conv_template.sep.strip())[0].strip().rstrip('.').strip() for response in responses]

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
        print(f'DEBUG Responses {responses}')
        print(f'DEBUG batch_confidences {batch_confidences}')

        return responses, batch_confidences

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def _single_frame_worker(frame, max_num, input_size):
    """Worker function for parallel tiling on CPU."""
    # We must rebuild the transform inside the worker for pickling
    transform = build_transform(input_size=input_size)
    tiles = dynamic_preprocess(frame, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_tensors = torch.stack([transform(tile) for tile in tiles])
    return pixel_tensors, pixel_tensors.shape[0]

def build_transform(input_size: int) -> T.Compose:
    """Builds a transformation pipeline for the given input size."""
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
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