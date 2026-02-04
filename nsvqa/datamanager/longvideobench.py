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

class LongVideoBench(Manager):
    def __init__(self, dataset_path, burned_path, categories, postprocess_dir="", read_number = 1000):
        self._dataset_path = dataset_path
        self._burned_path = burned_path
        if postprocess_dir:
            self.postprocess_dir = postprocess_dir
        if not isinstance(categories, list):
            self._categories = [categories]
        else:
            self._categories = categories
            
        self.read_number = read_number

    def load_data(self):
        category_buckets = defaultdict(list)

        with open(os.path.join(self._dataset_path, "lvb_val.json"), 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            for item in dataset:
                cat = item["question_category"]
                if cat in self._categories and len(category_buckets[cat]) < self.read_number:
                    video_path = os.path.join(self._burned_path, f"{item['video_path']}")
                    if not os.path.exists(video_path):
                        print(f"Burnt Video Does Not Exist: {video_path}")
                        continue
                    
                    entry = {
                        "question": item["question"],
                        "candidates": item["candidates"],
                        "correct_choice": item["correct_choice"],
                        "paths": {
                            "video_path": video_path,
                            "raw_video_path": os.path.join(self._dataset_path, "videos", item["video_path"]),
                            "subtitle_path": os.path.join(self._dataset_path, "subtitles", item["subtitle_path"])
                        },
                        "metadata": {
                            "video_id": item["video_id"],
                            "id": item["id"],
                            "position": item["position"],
                            "question_wo_referring_query": item["question_wo_referring_query"],
                            "topic_category": item["topic_category"],
                            "question_category": cat,
                            "level": item["level"],
                            "duration_group": item["duration_group"],
                            "starting_timestamp_for_subtitles": item["starting_timestamp_for_subtitles"],
                            "duration": item["duration"],
                            "view_count": item["view_count"],
                        }
                    }

                    category_buckets[cat].append(entry)

        # Flatten list of all selected entries from each category
        return [entry for entries in category_buckets.values() for entry in entries]


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