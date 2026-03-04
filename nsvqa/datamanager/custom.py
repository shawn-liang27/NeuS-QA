from nsvqa.datamanager.manager import Manager

from collections import defaultdict
from tqdm import tqdm
import hashlib
import shutil
import json
import copy
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def _crop_worker_static(cls_instance, entry, measure_metrics):
    start = time.perf_counter() if measure_metrics else 0
    try:
        # We pass the instance so we can still call crop_video
        cls_instance.crop_video(
            entry,
            save_path=entry["paths"]["cropped_path"],
            ground_truth=False
        )
        success = os.path.exists(entry["paths"]["cropped_path"])
        duration = time.perf_counter() - start if measure_metrics else 0
        return entry, success, duration
    except Exception as e:
        print(f"Error cropping {entry.get('metadata', {}).get('id')}: {e}")
        return entry, False, 0

class Custom(Manager):
    def __init__(self, burned_dir,raw_data=None, dataset_path=None, postprocess_dir=None):
        self.raw_data = raw_data
        self.postprocess_dir = postprocess_dir
        self._burned_dir = burned_dir
        self._dataset_path = dataset_path
    def load_data(self) -> list:
        assert self.raw_data is not None
        ret = []
        for raw_entry in self.raw_data:
            # ret.append({
            #     "paths": {
            #         "video_path": raw_entry["video_path"]
            #     },
            #     "metadata": {"video_id" : raw_entry["video_id"]},
            #     "question": raw_entry["question"],
            #     "candidates": raw_entry["answer_choices"],
            # })
            video_path = os.path.join(self._burned_dir, f"{raw_entry['video_path']}")
            entry = {
                    "question": raw_entry["question"],
                    "candidates": raw_entry["candidates"],
                    "correct_choice": raw_entry["correct_choice"],
                    "paths": {
                        "video_path": video_path,
                        "raw_video_path": os.path.join(self._dataset_path, "videos", raw_entry["video_path"]),
                        "subtitle_path": os.path.join(self._dataset_path, "subtitles", raw_entry["subtitle_path"])
                    },
                    "metadata": {
                        "video_id": raw_entry["video_id"],
                        "id": raw_entry["id"],
                        "position": raw_entry["position"],
                        "question_wo_referring_query": raw_entry["question_wo_referring_query"],
                        "topic_category": raw_entry["topic_category"],
                        "question_category": raw_entry["question_category"],
                        "level": raw_entry["level"],
                        "duration_group": raw_entry["duration_group"],
                        "starting_timestamp_for_subtitles": raw_entry["starting_timestamp_for_subtitles"],
                        "duration": raw_entry["duration"],
                        "view_count": raw_entry["view_count"],
                    }
                }
            ret.append(entry)
        return ret

        
    def postprocess_data(self, nsvs_path, measure_metrics=False):
        assert self.postprocess_dir is not None
        self._nsvs_path = nsvs_path

        cropped_dir = os.path.join(os.path.dirname(self.postprocess_dir), "cropped_videos")
        os.makedirs(cropped_dir, exist_ok=True)

        with open(self._nsvs_path, "r") as f:
            nsvs_data = json.load(f)

        for entry in nsvs_data:
            entry["paths"]["cropped_path"] = os.path.join(cropped_dir, f'{entry["metadata"]["id"]}.mp4')

        output = []
    
        max_workers = 10
        print(f"Starting parallel cropping with {max_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Map the tasks to the executor
            futures = {
                executor.submit(_crop_worker_static, self, entry, measure_metrics): entry 
                for entry in nsvs_data
            }
            
            # tqdm tracks progress as tasks complete
            for future in tqdm(as_completed(futures), total=len(nsvs_data), desc="Cropping Videos"):
                entry, success, duration = future.result()
                
                if success:
                    if measure_metrics:
                        # Initialize time_metrics if it doesn't exist
                        entry.setdefault("time_metrics", {})["cropping_video_time"] = duration
                    output.append(entry)

        # 4. Save results
        with open(self.postprocess_dir, "w") as f:
            json.dump(output, f, indent=4)
            
