import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd

def get_dataframe(raw_results, num_of_frame_in_sequence, out_dir):
    """
    raw_results: list of { "category": str, "time_metrics": dict }
    time_metrics dict format:
    time_metrics: {
        "completion_time" : float,
        "PULS_time" : float,
        "Target_ID_time" : float, 
        "NeuS_time" : float,
        "num_propositions" : int,
        "frame_count" : int,
        "video_IO_time" : float
        "num_frame_windows" : int,
        "per_proposition_detection_time" : list[float],
        "per_frame_window_detection_time" : list[float],
        "model_checks_time" : list[float],
        "num_model_checks" : int,
        "num_vlm_detections" : int
    """
    metric_list = []
    for result in raw_results:
        metrics = result["time_metrics"]
        metrics["id"] = result["metadata"]["id"]
        metrics["question_category"] = result["metadata"]["question_category"]
        metrics["duration_group"] = result["metadata"]["duration_group"]
        metrics["fps"] = result["metadata"]["fps"]
        metrics["duration"] = result["metadata"]["duration"]
        metrics["num_of_frame_in_sequence"] = num_of_frame_in_sequence
        metric_list.append(metrics)
    df = pd.DataFrame(metric_list)
    df.to_csv(f'{out_dir}/metric_data.csv', index=False)

def aggregate_metrics(raw_results):
    """
    raw_results: list of { "category": str, "time_metrics": dict }
    time_metrics dict format:
    time_metrics: {
        "completion_time" : float,
        "PULS_time" : float,
        "Target_ID_time" : float, 
        "NeuS_time" : float,
        "num_propositions" : int,
        "frame_count" : int,
        "video_IO_time" : float
        "num_frame_windows" : int,
        "per_proposition_detection_time" : list[float],
        "per_frame_window_detection_time" : list[float],
        "model_checks_time" : list[float],
        "num_model_checks" : int,
        "num_vlm_detections" : int
    }
    """
    # 1. Initialize data structures
    category_accumulators = defaultdict(lambda: defaultdict(float))
    category_counts = defaultdict(int)
    
    global_accumulator = defaultdict(float)
    global_count = 0

    # List of keys to sum directly
    scalar_keys = [
        "completion_time", "PULS_time", "Target_ID_time", "NeuS_time", "frame_count", 
        "num_propositions", "num_frame_windows", "num_model_checks", "num_vlm_detections", "video_IO_time", "automaton_set_up_time"
    ]
    # List of keys that are lists in raw data (require nested summing)
    list_keys = ["per_proposition_detection_time", "model_checks_time", "per_frame_window_detection_time"]

    # 2. Single-pass Accumulation
    for entry in raw_results:
        entry["time_metrics"]["frame_count"] = entry["metadata"].get("frame_count", 0)
        cat = entry["metadata"]["duration_group"]
        metrics = entry["time_metrics"]
        
        category_counts[cat] += 1
        global_count += 1

        for key in scalar_keys:
            val = metrics.get(key, 0)
            category_accumulators[cat][key] += val
            global_accumulator[key] += val

        for key in list_keys:
            val_list = metrics.get(key, [])
            length = len(val_list) if len(val_list) > 0 else 1
            total_time = sum(val_list) / length
            # We track the sum of sums for the high-level average
            category_accumulators[cat][key] += total_time
            global_accumulator[key] += total_time

    # 3. Final Aggregation
    category_aggregated = {}
    for cat, totals in category_accumulators.items():
        count = category_counts[cat]
        category_aggregated[cat] = {
            "count": count,
            **{f"avg_{k}": v / count for k, v in totals.items()}
        }

    global_aggregated = {
        "total_questions": global_count,
        **{f"global_avg_{k}": v / global_count for k, v in global_accumulator.items()}
    }

    return {"Categorical Metrics" : category_aggregated, "Global Metrics" :global_aggregated}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nsvs_result_dir")
    parser.add_argument("frames_window", type=int, default=3)
    args = parser.parse_args()
    nsvs_result_dir = args.nsvs_result_dir
    out_file = f'{nsvs_result_dir}/run_time_metrics.json'

    raw_res = []
    p = Path(nsvs_result_dir)
    found_files = p.rglob(f'**/postprocess_output_*.json')
    for file_path in found_files:
        with open(file_path, "r") as file:
            out = json.load(file)
            raw_res.extend(out)
    
    metrics = aggregate_metrics(raw_res)
    get_dataframe(raw_res, args.frames_window, nsvs_result_dir)
    print(metrics)

    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f'Aggregated Result saved at {out_file}')
if __name__ == "__main__":
    main()