from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import numpy as np
import base64
import json
import tqdm
import cv2
import os
from collections import defaultdict
from nsvqa.vqa.parse_response import parse_multi_choice_response
from decord import VideoReader, cpu

NUM_WORKERS = 4

class VLLMClient:
    def __init__(
        self,
        api_key="EMPTY",
        api_base="http://localhost:8000/v1",
        model="OpenGVLab/InternVL3_5-14B",
    ):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
        # 1. Get the directory where this script (lmm.py) is located
        _script_dir = os.path.dirname(os.path.abspath(__file__))

        # 2. Join it with the filename
        _file_path = os.path.join(_script_dir, "SYSTEM_PROMPTS.json")
        with open(_file_path, "r") as file:
            _system_prompt = json.load(file)
        self.system_prompt = _system_prompt["default"]

    def _encode_frame(self, frame):
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            raise ValueError("Could not encode frame")
        return base64.b64encode(buffer).decode("utf-8")

    def multiple_choice(self, frames_by_cam: dict, question: str, candidates: list[str]) -> str:
        user_content = []
        frames = list(frames_by_cam.values())[0]
        encoded_images = [self._encode_frame(frame) for frame in frames]
        for i, encoded in enumerate(encoded_images):
            # user_content.append({
            #     "type": "text", 
            #     "text": f"Frame{i+1}: "
            # })
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}
            })
            # user_content.append({
            #     "type": "text", 
            #     "text": "\n"
            # })

        parsing_rule = "\nAnswer with the option's letter from the given choices directly."
        prompt = f"{question}\n"
        for candidate in candidates:
            prompt += f"{candidate}\n"
        prompt += f"{parsing_rule}"
        user_content.append({"type": "text", "text": prompt})
        # print("="*50)
        # print(user_content)
        # print("="*50)
        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                # {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
        )
        return chat_response.choices[0].message.content

class VLLMClientMultiprocessing(VLLMClient):
    def __init__(
        self,
        model,
        api_base,
        max_workers=NUM_WORKERS,
    ):
        super().__init__(model=model, api_base=api_base)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def multiple_choice_batch(self, batch_args):
        futures = [
            self.executor.submit(self.multiple_choice, *args) for args in batch_args
        ]
        
        results = []
        for future in tqdm.tqdm(futures, desc="Processing batch"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing a task: {e}")
                results.append(None)
      
        return results

def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return frame_count

def load_video_frames(video_path, num_frames):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        real_len = len(vr) - 1 
        
        if real_len < num_frames:
            indices = np.arange(max(1, real_len))
        else:
            indices = np.linspace(0, real_len - 1, num_frames).astype(int)
            
        indices = np.clip(indices, 0, len(vr) - 1)
        
        frames = vr.get_batch(indices).asnumpy()
        return [frame for frame in frames]
    except Exception as e:
        print(f"Warning: Decord failed for {video_path} ({str(e)}). Fallback to OpenCV.")
        
        # 2. Fallback to OPENCV (Best Stability)
        # If Decord crashes, OpenCV usually handles corrupt footers gracefully
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < num_frames:
            indices = np.arange(total_frames)
        else:
            indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
            
        images = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # OpenCV returns BGR, convert to RGB to match Decord/Model expectation
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(frame)
            else:
                # If we can't read a frame, just pad with the last valid one or skip
                if images:
                    images.append(images[-1])
        
        cap.release()
        return images

def run_experiment(data, vllm_client, output_path, max_num_frames=32, eval=True, pure_vqa=False):
    results = []
    batch_args_all_calls = []
    
    for entry in data:
        if pure_vqa:
            vid_path = entry["paths"]["video_path"]
            print(f"Flag pure_vqa is set to {pure_vqa}, using the raw video for vqa {vid_path}")
        else:
            vid_path = entry["paths"]["cropped_path"]
            print(f"Flag pure_vqa is set to {pure_vqa}, using the cropped video processed by nsvs for vqa {vid_path}")
        frames = load_video_frames(vid_path, num_frames=max_num_frames)
        if not frames:
            continue
        for i in range(len(entry["candidates"])):
            entry["candidates"][i] = f"{chr(65+i)}. {entry['candidates'][i]}"
        batch_args_all_calls.append(({"main": frames}, entry["question"], entry["candidates"]))

    predicted_answers_all_calls = vllm_client.multiple_choice_batch(batch_args_all_calls)
    total_correct = 0
    vqa_res = defaultdict(dict)
    for i, entry in enumerate(data):
        raw_pred = predicted_answers_all_calls[i]

        parsed_pred = parse_multi_choice_response(raw_pred, len(entry["candidates"]))

        output_dict = {
            "video_path": entry["paths"]["video_path"] if pure_vqa else entry["paths"]["cropped_path"],
            "question": entry["question"],
            "candidates": entry["candidates"],
            "raw_prediction" : raw_pred,
            "parsed_prediction": parsed_pred,
            "question_category" : entry.get("metadata", {}).get("question_category", "unknown"), 
            "video_id" : entry["metadata"]["video_id"],
            "id" : entry["metadata"]["id"]
        }
        
        if eval:
            correct_answer_letter = chr(65 + entry["correct_choice"])
            
            is_correct = 1 if parsed_pred == correct_answer_letter else 0
            total_correct += is_correct
            
            output_dict["correct_answer"] = correct_answer_letter
            output_dict["is_correct"] = is_correct

            category = entry.get("metadata", {}).get("question_category", "uncategorized")
            vqa_res[category]["total"] = vqa_res[category].get("total",0) + 1
            vqa_res[category]["num_correct"] = vqa_res[category].get("num_correct",0) + is_correct
            
        results.append(output_dict)

    if eval and len(data) > 0:
        for category in vqa_res.keys():
            vqa_res[category]["accuracy"] = vqa_res[category]["num_correct"] / vqa_res[category]["total"]
        accuracy = total_correct / len(data)
        print(f"Total Accuracy: {accuracy:.2%}")

        with open(os.path.join(os.path.dirname(output_path), "vqa_summary.json"), "w") as f:
            json.dump(vqa_res, f, indent=4)
    else:
        for entry in results:
            print(entry["predicted_answer"])
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    

def lmm_eval_vqa(dataset_path, output_path, vlm_config, max_num_frames, eval=True, pure_vqa=False):
    if pure_vqa:
        data = dataset_path
    else:
        with open(dataset_path, "r") as f:
            data = json.load(f)

    vllm_client = VLLMClientMultiprocessing(
        model=vlm_config[1],
        api_base=f"http://localhost:{vlm_config[0]}/v1"
    )
    run_experiment(
        data=data,
        vllm_client=vllm_client,
        output_path=output_path,
        max_num_frames=max_num_frames,
        pure_vqa=pure_vqa, 
        eval=eval
    )

