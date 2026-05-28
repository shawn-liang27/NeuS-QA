from nsvqa.datamanager.manager import Manager

from collections import defaultdict
from tqdm import tqdm
import hashlib
import shutil
import json
import copy
import time
import os

class VideoMME(Manager):
    def __init__(self, dataset_path, burned_path, postprocess_dir, categories=None, read_number=1000):
        super().__init__()
        self._dataset_path = dataset_path
        self._burned_path = burned_path
        self.read_number = read_number
        self.postprocess_dir = postprocess_dir
        self._categories = categories if isinstance(categories, list) else ([categories] if categories else [])

    def load_data(self):
        def clean_circled_numbers(text):
            # Mapping Unicode circled numbers to standard ASCII numbers
            mapping = str.maketrans({
                '①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5',
                '⑥': '6', '⑦': '7', '⑧': '8', '⑨': '9', '⑩': '10'
            })
            return text.translate(mapping)

        category_buckets = defaultdict(list)
        # Video-MME usually has a 'test.json'
        data_json = os.path.join(self._burned_path, "dataset.json")
        
        # Path to the unzipped subtitles folder
        subtitle_root = os.path.join(self._dataset_path, "subtitle")

        with open(data_json, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            for item in dataset:
                duration_group = item.get("duration", "unknown")
                video_id = item['videoID']
                video_filename = f"{video_id}.mp4"
                question_id = item["question_id"]
                if video_filename == "IKtnfFHjERg.mp4":
                    continue
                # Check for subtitle file (.srt is standard for Video-MME)
                subtitle_file = os.path.join(subtitle_root, f"{video_id}.srt")
                subtitle_path = subtitle_file if os.path.exists(subtitle_file) else None
                item["video_path"] = os.path.join(self._burned_path, duration_group, video_filename)
                
                entry = {
                    "question": item["question"],
                    "candidates": item["options"],
                    "correct_choice": item["answer"],
                    "paths": {
                        "video_path": os.path.join(self._burned_path, duration_group, video_filename),
                        "raw_video_path": os.path.join(self._dataset_path, "video", duration_group, video_filename),
                        "subtitle_path": subtitle_path,
                        "url": item["url"], 
                    },
                    "metadata": {
                        "video_id": video_id,
                        "question_id": question_id,
                        "id": f"{video_id}_{question_id}",
                        "duration_group": duration_group,
                        "domain": item["domain"],
                        "sub_category": item["sub_category"],
                        "task_type": item["task_type"],
                    }
                }
                entry["question"] = clean_circled_numbers(entry["question"])
                entry['candidates'] = [clean_circled_numbers(c) for c in entry['candidates']]
                
                category_buckets[duration_group].append(entry)
        return [entry for entries in category_buckets.values() for entry in entries]

    def postprocess_data(self, nsvs_path, measure_metrics=True, postprocess_dir="", ):
        assert self.postprocess_dir is not None
        self._nsvs_path = nsvs_path

        cropped_dir = os.path.join(os.path.dirname(self.postprocess_dir), "cropped_videos")
        os.makedirs(cropped_dir, exist_ok=True)

        with open(self._nsvs_path, "r") as f:
            nsvs_data = json.load(f)

        output = []
        for entry_nsvs in tqdm(nsvs_data):
            
            entry_nsvs["paths"]["cropped_path"] = os.path.join(cropped_dir, f'{entry_nsvs["metadata"]["id"]}.mp4')
            crop_start = time.perf_counter() if measure_metrics else 0
            self.crop_video(
                entry_nsvs,
                save_path=entry_nsvs["paths"]["cropped_path"],
                ground_truth=False
            )
            if measure_metrics: 
                entry_nsvs["time_metrics"]["cropping_video"] = time.perf_counter() - crop_start

            if os.path.exists(entry_nsvs["paths"]["cropped_path"]): # if crop successful
                output.append(entry_nsvs)

        with open(self.postprocess_dir, "w") as f:
            json.dump(output, f, indent=4)