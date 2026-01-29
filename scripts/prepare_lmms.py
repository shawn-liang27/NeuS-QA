import json
import os
import shutil
import argparse
from pathlib import Path

def main(args):
    longvideobench_path = "/usr/homes/sgl57/.data/LongVideoBench/lvb_val.json"
    nsvs_path = args.nsvs_path
    logging_path = args.logging_path
    lmms_path = os.path.join(logging_path, "lmms")
    postprocess_path_list = steps = [os.path.join(nsvs_path, f'split_{i}', 'postprocess_output') for i in range(1, 5)]
    print(postprocess_path_list)
    os.makedirs(lmms_path, exist_ok=True)
    original_output_path = os.path.join(lmms_path, "original")
    neusqa_output_path = os.path.join(lmms_path, "neusqa")
    os.makedirs(original_output_path, exist_ok=True)
    os.makedirs(neusqa_output_path, exist_ok=True)

    with open(longvideobench_path, "r") as f:
        original_data = json.load(f)

    postprocess_data = []
    output_files = []

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
        original_map[(original_entry["question"], original_entry["video_id"])] = original_entry

    original_output_data = []
    neusqa_output_data = []
    
    videos_to_move = set()

    for postprocess_entry in postprocess_data:
        key = (postprocess_entry["question"], postprocess_entry["metadata"]["video_id"])
        if key in original_map:
            original_entry = original_map[key]
            new_entry = original_entry.copy()
            cropped_path = postprocess_entry["paths"]["cropped_path"]

            filename = os.path.basename(cropped_path)
            stem = os.path.splitext(filename)[0]
            new_entry["video_id"] = stem
            new_entry["id"] = stem
            new_entry["video_path"] = filename
            
            original_output_data.append(original_entry)
            neusqa_output_data.append(new_entry)

            videos_to_move.add(original_entry["video_path"])

    for entry in original_output_data:
        for i, candidate in enumerate(entry["candidates"]):
            entry[f"option{i}"] = candidate
    for entry in neusqa_output_data:
        for i, candidate in enumerate(entry["candidates"]):
            entry[f"option{i}"] = candidate

    with open(os.path.join(original_output_path, "lvb_val.json"), "w") as f:
        json.dump(original_output_data, f, indent=4)
    with open(os.path.join(neusqa_output_path, "lvb_val.json"), "w") as f:
        json.dump(neusqa_output_data, f, indent=4)

    
    # original videos
    burn_subtitles_path = "/usr/homes/sgl57/.data/LongVideoBench/burn-subtitles/T3E_E3E_T3O_O3O_mix_2026_01_14_21_55"
    original_output_videos_path = os.path.join(original_output_path, "videos")
    os.makedirs(original_output_videos_path, exist_ok=True)

    print(f"Burned Videos to move: {len(videos_to_move)}")
    for video_path in videos_to_move:
        burned_subtitles_video_file = os.path.join(burn_subtitles_path, video_path)
        original_output_videos_file = os.path.join(original_output_videos_path, video_path)
        shutil.copy(burned_subtitles_video_file, original_output_videos_file)

    # neusqa videos

    neusqa_videos_output_path = os.path.join(neusqa_output_path, "videos")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logging_path")
    parser.add_argument("--nsvs_path")
    args = parser.parse_args()
    main(args)
