import json
import os
import shutil
import argparse
from pathlib import Path


def main(args):
    data_dir = "/usr/homes/sgl57/.data/mlvu/MLVU"
    if args.task_type == "original":
        new_original_entries = []
        with open("/usr/homes/sgl57/.data/mlvu/MLVU/dataset.json", "r") as f:
            dataset_json = json.load(f)
            for original_entry in dataset_json:
                entry = {}
                entry["video"] = original_entry["metadata"]["video_id"]
                entry["video_path"] = original_entry["paths"]["video_path"]
                entry["duration"] = original_entry["metadata"]["duration"]
                entry["question"] = original_entry["question"]
                entry["candidates"] = original_entry["candidates"]
                entry["answer"] = original_entry["correct_choice"]
                entry["question_type"] = original_entry["metadata"]["question_type"]
                entry['frames_of_interest'] = [[-1]]
                new_original_entries.append(entry)
            
        original_dataset_file = os.path.join(data_dir, f"original_dataset_lmms.json")
        with open(original_dataset_file, "w") as f:
            json.dump(new_original_entries, f, indent=4)
        print(f'[DEBUG] Final original data dir size {len(new_original_entries)}')
        print(f'[DEBUG] Data File stored at {original_dataset_file}')
        return
    
    nsvs_path = args.nsvs_path
    logging_path = f'/usr/homes/sgl57/NeuS-VLM/NeuS-QA/experiment_results/lmm_eval/mlvu/{args.task_type}/experiment_{args.experiment_number}'
    lmms_path = os.path.join(logging_path, "lmms")
    nsvqa_path_list = [os.path.join(nsvs_path, f'split_{i}', 'nsvqa_output') for i in range(1, args.num_split + 1)]
    os.makedirs(lmms_path, exist_ok=True)

    neusqa_output_path = os.path.join(lmms_path, "neusqa")

    os.makedirs(neusqa_output_path, exist_ok=True)
    os.makedirs(f'{data_dir}/{args.task_type}', exist_ok=True)
    
    postprocess_data = []
    output_files = []

    # Find all prostprocess_output_*.json cross splits
    p = Path(nsvs_path)
    found_files = p.rglob(f'nsvqa_output_*.json')
    for file_path in found_files:
        print(file_path)
        file_idx = int(str(file_path).split('_')[-1].split('.')[0])
        output_files.append((file_idx, os.path.join(nsvqa_path_list[file_idx - 1], file_path)))

    print(output_files)
    output_files.sort(key=lambda x: x[0])
    for _, file_path in output_files:
        with open(file_path, "r") as f:
            data = json.load(f)
            postprocess_data.extend(data)

    print(f'Length of Postprocessed Data: {len(postprocess_data)}')

    output_data = []
    for postprocess_entry in postprocess_data:
        entry = {}
        entry["video"] = postprocess_entry["metadata"]["video_id"]
        entry["video_path"] = postprocess_entry["paths"]["video_path"]
        entry["duration"] = postprocess_entry["metadata"]["duration"]
        entry["question"] = postprocess_entry["question"]
        entry["candidates"] = postprocess_entry["candidates"]
        entry["answer"] = postprocess_entry["correct_choice"]
        entry["question_type"] = postprocess_entry["metadata"]["question_type"]

        frames_of_interest = postprocess_entry["frames_of_interest"]

        if frames_of_interest == [-1]:
            entry['frames_of_interest'] = [[-1]]
        else:
            entry['frames_of_interest'] = frames_of_interest

        
        output_data.append(entry)

    output_data_file_path = os.path.join(data_dir, args.task_type, f"experiment_{args.experiment_number}.json")
    with open(output_data_file_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f'[DEBUG] Final neusqa data dir size {len(output_data)}')
    print(f'[DEBUG] Data File stored at {output_data_file_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_number")
    parser.add_argument("--nsvs_path")
    parser.add_argument("--num_split", type=int, default=8)
    parser.add_argument("--task_type", type=str, default="rt-neus")
    args = parser.parse_args()
    main(args)
