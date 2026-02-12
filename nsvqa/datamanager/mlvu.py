from nsvqa.datamanager.manager import Manager

from collections import defaultdict
from tqdm import tqdm
import hashlib
import shutil
import json
import copy
import time
import os

CATEGORY_DATASET_MAP={
    "2_needle": "json/2_needle.json",
    "3_ego": "json/3_ego.json"
}

class MLVU(Manager):
    def __init__(self, dataset_path, postprocess_dir, burned_path=None, categories=None, read_number=1000):
        super().__init__()
        self._dataset_path = dataset_path
        self._burned_path = burned_path
        self.read_number = read_number
        self.postprocess_dir = postprocess_dir
        self._categories = categories if isinstance(categories, list) else ([categories] if categories else [])

    def load_data(self):
        full_dataset = []
        for category in self._categories:
            base_data_file = CATEGORY_DATASET_MAP.get(category, "")
            if base_data_file:
                data_file = os.path.join(self._dataset_path, base_data_file)
                with open(data_file, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
                    for item in dataset:
                        video_id = item['video'].split(".mp4")[0]
                        item["id"] = video_id

                        video_full_path = os.path.join(self._dataset_path, "video", category ,item["video"])
                        
                        entry = {
                            "question": item["question"],
                            "candidates": item["candidates"],
                            "correct_choice": item["answer"],
                            "paths": {
                                "video_path": video_full_path,
                            },
                            "metadata": {
                                "video_id": item['video'],
                                "id": f"{video_id}",
                                "duration": item["duration"],
                                "question_type": item["question_type"],
                            }
                        }
                        full_dataset.append(entry)
        with open(f'{self._dataset_path}/dataset.json', 'w') as f:
            json.dump(full_dataset, f, indent=4)
        
        return full_dataset

    def postprocess_data(self, nsvs_path, postprocess_dir="", ):
        assert self.postprocess_dir is not None
        self._nsvs_path = nsvs_path

        cropped_dir = os.path.join(os.path.dirname(self.postprocess_dir), "cropped_videos")
        os.makedirs(cropped_dir, exist_ok=True)

        with open(self._nsvs_path, "r") as f:
            nsvs_data = json.load(f)

        output = []
        for entry_nsvs in tqdm(nsvs_data):
            
            entry_nsvs["paths"]["cropped_path"] = os.path.join(cropped_dir, f'{entry_nsvs["metadata"]["id"]}.mp4')
            self.crop_video(
                entry_nsvs,
                save_path=entry_nsvs["paths"]["cropped_path"],
                ground_truth=False
            )
            
            if os.path.exists(entry_nsvs["paths"]["cropped_path"]): # if crop successful
                output.append(entry_nsvs)

        with open(self.postprocess_dir, "w") as f:
            json.dump(output, f, indent=4)