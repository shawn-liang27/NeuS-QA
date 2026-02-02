from nsvqa.datamanager.manager import Manager

from collections import defaultdict
from tqdm import tqdm
import hashlib
import shutil
import json
import copy
import os
import time

class LongVideoBench(Manager):
    def __init__(self, dataset_path, burned_path, categories, postprocess_dir="", read_number = 1000):
        self._dataset_path = dataset_path
        self._burned_path = burned_path
        if postprocess_dir:
            self.postprocess_dir = postprocess_dir
        # self._categories = ["S2E", "S2O", "S2A", "E2O", "T2A", "T2E", "T2O", "O2E", "SSS", "TAA", "E3E", "SAA", "T3O", "TOS", "O3O", "SOS", "T3E"]
        if not isinstance(categories, list):
            self._categories = [categories]
        else:
            self._categories = categories
            
        self.read_number = read_number # max reads per category

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

        # run_name = self._nsvs_path.split('/')[-1].split('.')[0].replace('longvideobench_', '')
        # self._output_path_nsvqa = f"/nas/mars/experiment_result/nsvqa/6_formatted_output/longvideobench_nsvqa_{run_name}"
        # if self.compile_position:
        #     self._output_path_position = f"/nas/mars/experiment_result/nsvqa/6_formatted_output/longvideobench_position_{run_name}"
        # if self.compile_full:
        #     self._output_path_full = f"/nas/mars/experiment_result/nsvqa/6_formatted_output/longvideobench_full_{run_name}"

        # os.makedirs(os.path.join(self._output_path_nsvqa, "videos"), exist_ok=True)
        
        # if self.compile_position:
        #     os.makedirs(os.path.join(self._output_path_position, "videos"), exist_ok=True)
        # if self.compile_full:
        #     os.makedirs(os.path.join(self._output_path_full, "videos"), exist_ok=True)
        #     shutil.copytree("/nas/mars/experiment_result/nsvqa/6_formatted_output/longvideobench_full/video", os.path.join(self._output_path_full, "videos"), dirs_exist_ok=True)

        # with open(os.path.join(self._dataset_path, "lvb_val.json"), "r") as f:
        #     lvb_data = json.load(f)
        # with open(self._nsvs_path, "r") as f:
        #     nsvs_data = json.load(f)

        # output_nsvqa = []    # nsvqa cropped video
        # output_full = []     # entire video
        # for entry_nsvs in tqdm(nsvs_data):
        #     found = False
        #     for entry in lvb_data:
        #         if entry["question"] == entry_nsvs["question"] and entry["id"] == entry_nsvs["metadata"]["id"]:
        #             found = True

        #             candidates = entry["candidates"]
        #             for i in range(5):
        #                 if i < len(candidates):
        #                     entry[f"option{i}"] = candidates[i]
        #                 else:
        #                     entry[f"option{i}"] = "N/A"

        #             entry_full = copy.deepcopy(entry)

        #             code = entry["question"] + entry["id"]
        #             id = hashlib.sha256(code.encode()).hexdigest()
        #             entry["id"] = id + "_0"
        #             entry["video_id"] = id
        #             entry["paths"]["video_path"] = id + ".mp4"


        #             self.crop_video(
        #                 entry_nsvs, 
        #                 save_path=os.path.join(self._output_path_nsvqa, "videos", entry["paths"]["video_path"]),
        #                 ground_truth=False
        #             )

        #             if os.path.exists(os.path.join(self._output_path_nsvqa, "videos", entry["paths"]["video_path"])): # if crop is successful
        #                 if self.compile_position:
        #                     self.crop_video(
        #                         entry_nsvs, 
        #                         save_path=os.path.join(self._output_path_position, "videos", entry["paths"]["video_path"]),
        #                         ground_truth=True
        #                     )

        #                 output_nsvqa.append(entry)
        #                 output_full.append(entry_full)

        #     if found == False:
        #         print(f"Entry not found for question: {entry_nsvs['question']}")

        # with open(os.path.join(self._output_path_nsvqa, "lvb_val.json"), "w") as f:
        #     json.dump(output_nsvqa, f, indent=4)
        # if self.compile_position:
        #     with open(os.path.join(self._output_path_position, "lvb_val.json"), "w") as f:
        #         json.dump(output_nsvqa, f, indent=4)
        # if self.compile_full:
        #     with open(os.path.join(self._output_path_full, "lvb_val.json"), "w") as f:
        #         json.dump(output_full, f, indent=4)


# 0   to 53
# 54  to 107
# 108 to 161
# 162 to 215
# 216 to 270


# 0 to 67
# 68 to 135
# 136 to 202
# 202 to 270



# 0 to 92
# 93 to 185
# 186 to 278
# 279 to 372
# 373 to 466
# 467 to 560
# 561 to 653
# 654 to 747

