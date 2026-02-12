import json
import os
import shutil
import argparse
from pathlib import Path

COPY_ORIGINAL=False

def main(args):
    video_mme_path = "/usr/homes/sgl57/.data/Video-MME/burn-subtitles/dataset.json"
    nsvs_path = args.nsvs_path
    logging_path = f'/usr/homes/sgl57/NeuS-VLM/NeuS-QA/experiment_results/lmm_eval/video_mme/{args.task_type}/experiment_{args.experiment_number}'
    lmms_path = os.path.join(logging_path, "lmms")
    postprocess_path_list = [os.path.join(nsvs_path, f'split_{i}', 'postprocess_output') for i in range(1, args.num_split + 1)]
    print(postprocess_path_list)
    os.makedirs(lmms_path, exist_ok=True)
    neusqa_output_path = os.path.join(lmms_path, "neusqa")

    os.makedirs(neusqa_output_path, exist_ok=True)

    # Read original dataset json file
    with open(video_mme_path, "r") as f:
        original_data = json.load(f)

    postprocess_data = []
    output_files = []

    # Find all prostprocess_output_*.json cross splits
    p = Path(nsvs_path)
    found_files = p.rglob(f'postprocess_output_*.json')
    for file_path in found_files:
        print(file_path)
        file_idx = int(str(file_path).split('_')[-1].split('.')[0])
        output_files.append((file_idx, os.path.join(postprocess_path_list[file_idx - 1], file_path)))

    print(output_files)
    output_files.sort(key=lambda x: x[0])
    for _, file_path in output_files:
        with open(file_path, "r") as f:
            data = json.load(f)
            postprocess_data.extend(data)

    print(f'Length of Postprocessed Data: {len(postprocess_data)}')
    original_map = {}
    for original_entry in original_data:
        original_map[(original_entry["question_id"], original_entry["videoID"])] = original_entry
    print(len(original_map))
    original_output_data = []
    neusqa_output_data = []
    
    videos_to_move = set()
    neusqa_subtitle_move = set()
    added_keys = []
    for postprocess_entry in postprocess_data:
        key = (postprocess_entry["metadata"]["question_id"], postprocess_entry["metadata"]["video_id"])
        if key in original_map:
            original_entry = original_map[key]
            new_entry = original_entry.copy()
            cropped_path = postprocess_entry["paths"]["cropped_path"]

            filename = os.path.basename(cropped_path)
            stem = os.path.splitext(filename)[0]
            new_entry["videoID"] = stem
            new_entry["id"] = stem
            new_entry["video_path"] = filename
            neusqa_subtitle_move.add((stem, original_entry["videoID"]))

            original_output_data.append(original_entry)
            neusqa_output_data.append(new_entry)
            videos_to_move.add((original_entry["video_path"], original_entry["videoID"]))
            added_keys.append(key)
    
    # for entry in original_output_data:
    #     for i, candidate in enumerate(entry["options"]):
    #         entry["options"].append()
    # for entry in neusqa_output_data:
    #     for i, candidate in enumerate(entry["candidates"]):
    #         entry[f"option{i}"] = candidate
    
    subtitle_dir = "/usr/homes/sgl57/.data/Video-MME/subtitle"

    neusqa_subtitle_output_path = os.path.join(neusqa_output_path, "subtitle")
    os.makedirs(neusqa_subtitle_output_path, exist_ok=True)

    for new_name, original_name in neusqa_subtitle_move:
        original_subtitle_path = os.path.join(subtitle_dir, f'{original_name}.srt')
        subtitle_file = os.path.join(neusqa_subtitle_output_path, f'{new_name}.srt')
        if os.path.exists(original_subtitle_path):
            shutil.copy(original_subtitle_path, subtitle_file)
        else:
            print(f"Subtitle {original_subtitle_path} not found")

    

    if COPY_ORIGINAL:
        original_output_path = os.path.join(lmms_path, "original")
        os.makedirs(original_output_path, exist_ok=True)
        original_output_subtitle_path = os.path.join(original_output_path, "subtitle")
        original_output_videos_path = os.path.join(original_output_path, "data")
        os.makedirs(original_output_videos_path, exist_ok=True)
        os.makedirs(original_output_subtitle_path, exist_ok=True)
        with open(os.path.join(original_output_path, "dataset.json"), "w") as f:
            json.dump(original_output_data, f, indent=4)
        print(f"Burned Videos to move: {len(videos_to_move)}")
        for video_path, videoID in videos_to_move:
            original_output_videos_file = os.path.join(original_output_videos_path, f'{videoID}.mp4')
            subtitle_file = os.path.join(subtitle_dir, f'{videoID}.srt')
            original_output_subtitle_file = os.path.join(original_output_subtitle_path, f'{videoID}.mp4')
            shutil.copy(video_path, original_output_videos_file)
            if os.path.exists(subtitle_file):
                shutil.copy(subtitle_file, original_output_subtitle_file)
            else:
                print(f"Subtitle {subtitle_file} not found")
    
    neusqa_videos_output_path = os.path.join(neusqa_output_path, "data")

    if os.path.exists(neusqa_videos_output_path):
        shutil.rmtree(neusqa_videos_output_path)

    for postprocess_path in postprocess_path_list:

        cropped_videos_dir = os.path.join(postprocess_path, "cropped_videos")

        print(cropped_videos_dir)
        source_dir = Path(cropped_videos_dir)
        target_dir = Path(neusqa_videos_output_path)

        # Ensure the target exists
        target_dir.mkdir(parents=True, exist_ok=True)

        for item in source_dir.iterdir():
            # Construct the full destination path
            dest_path = target_dir / item.name
            shutil.copy(item, dest_path)

    for original_key, original_entry in original_map.items():
        if original_key not in added_keys:
            new_entry = original_entry.copy()
            new_id = f"{new_entry["videoID"]}_{new_entry["question_id"]}"
            new_entry["videoID"] = new_id
            new_entry["id"] = new_id
            neusqa_output_data.append(new_entry)
            print(f"[DEBUG] Video {original_key} was not included, copying the original burned-in videos to /neusqa/data")
            target_dir = os.path.join(neusqa_videos_output_path, f'{new_id}.mp4')
            shutil.copy(original_entry["video_path"], target_dir)

    with open(os.path.join(neusqa_output_path, "dataset.json"), "w") as f:
        json.dump(neusqa_output_data, f, indent=4)
        
    print(f'[DEBUG] Final neusqa data dir size {len(neusqa_output_data)}')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_number")
    parser.add_argument("nsvs_path")
    parser.add_argument("num_split", type=int, default=8)
    parser.add_argument("--task_type", type=str, default="clip_relevancy_redundancy_batch_prop_dynamic_1")
    args = parser.parse_args()
    main(args)
