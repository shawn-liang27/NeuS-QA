import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd

BENCHMARK_NAME = {
    "video_mme": [
        "video_id",
        "question_id",
        "id",
        "duration_group",
        "domain",
        "sub_category",
        "task_type"
    ],
    "lvb": [
        "video_id",
        "id",
        "position",
        "question_wo_referring_query",
        "topic_category",
        "question_category",
        "level",
        "duration_group",
        "starting_timestamp_for_subtitles",
        "duration",
        "view_count"
    ]
}

def get_dataframe(raw_results, benchmark_name, out_dir):
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
    assert benchmark_name in BENCHMARK_NAME, f"'{benchmark_name}' is not a known benchmark"
    metadata_list = BENCHMARK_NAME[benchmark_name]
    metric_list = []

    for result in raw_results:
        metrics = result["time_metrics"]
        metrics["frames_of_interest"] = result["frames_of_interest"]
        for metadata in metadata_list:
            metrics[metadata] = result["metadata"][metadata]
        metric_list.append(metrics)
    df = pd.DataFrame(metric_list)
    df.to_csv(f'{out_dir}/metric_data.csv', index=False)

def aggregate_metrics(raw_results, is_naive):
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
        "num_propositions", "num_frame_windows", "num_model_checks", "num_vlm_detections", "video_IO_time", "automaton_set_up_time", "foi_count", "cropping_video_time", "pct_foi_out_of_full"
    ]
    # List of keys that are lists in raw data (require nested summing)
    list_keys = ["model_checks_time", "per_frame_window_detection_time"]

    # 2. Single-pass Accumulation
    for entry in raw_results:
        frame_count = entry["metadata"]["frame_count"]
        fps = entry["metadata"]["fps"]
        foi = entry["frames_of_interest"]

        total_foi = 0
        if is_naive:
            if foi == [-1]:
                total_foi = frame_count
            else:
                total_foi = foi[1] - foi[0]
        else:
            for segment in foi:
                if segment == -1:
                    diff = frame_count
                else:
                    diff = segment[1] - segment[0]
                total_foi += diff
            

        if not total_foi:
            total_foi = 1
        pct_foi_out_of_full = round(total_foi / frame_count, 5)

        entry["time_metrics"]["foi_count"] = total_foi
        entry["time_metrics"]["pct_foi_out_of_full"] = pct_foi_out_of_full
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

    # 5. Final Aggregation
    category_aggregated = {}
    for cat, totals in category_accumulators.items():
        count = category_counts[cat]
        category_aggregated[cat] = {
            "count": count,
            **{f"avg_{k}": v / count for k, v in totals.items()}
        }

        category_aggregated[cat][f'avg_foi_count'] = int(category_aggregated[cat][f'avg_foi_count'])
        category_aggregated[cat][f'avg_pct_foi_out_of_full'] = round(category_aggregated[cat][f'avg_pct_foi_out_of_full'], 5)
        category_aggregated[cat][f'avg_PULS_runtime_pct'] = round(category_aggregated[cat][f'avg_PULS_time'] / category_aggregated[cat][f'avg_completion_time'], 5)
        category_aggregated[cat][f'avg_target_id_runtime_pct'] = round(category_aggregated[cat][f'avg_Target_ID_time'] / category_aggregated[cat][f'avg_completion_time'], 5)
        category_aggregated[cat][f'avg_neus_runtime_pct'] = round(category_aggregated[cat][f'avg_NeuS_time'] / category_aggregated[cat][f'avg_completion_time'], 5)

        category_aggregated[cat][f'avg_cropping_video_time'] = round(category_aggregated[cat][f'avg_cropping_video_time'] / category_aggregated[cat][f'avg_completion_time'], 5)
        
        category_aggregated[cat][f'avg_model_checks_time_pct_out_neus'] = round(category_aggregated[cat][f'avg_model_checks_time'] / category_aggregated[cat][f'avg_NeuS_time'], 5)
        category_aggregated[cat][f'avg_model_checks_time_pct_out_neus'] = round(category_aggregated[cat][f'avg_per_frame_window_detection_time'] * category_aggregated[cat][f'avg_num_frame_windows'] / category_aggregated[cat][f'avg_NeuS_time'], 5)
        category_aggregated[cat][f'avg_model_checks_time_pct_out_neus'] = round(category_aggregated[cat][f'avg_cropping_video_time'] / category_aggregated[cat][f'avg_NeuS_time'], 5)


        category_aggregated[cat][f'avg_pct_foi_out_of_full'] = round(category_aggregated[cat][f'avg_pct_foi_out_of_full'], 5)


    global_aggregated = {
        "total_questions": global_count,
        **{f"global_avg_{k}": v / global_count for k, v in global_accumulator.items()}
    }
    print(global_aggregated)
    global_aggregated[f'global_avg_foi_count'] = int(global_aggregated[f'global_avg_foi_count'])
    global_aggregated[f'global_avg_pct_foi_out_of_full'] = round(global_aggregated[f'global_avg_pct_foi_out_of_full'], 5)
    global_aggregated[f'global_avg_PULS_runtime_pct'] = round(global_aggregated[f'global_avg_PULS_time'] / global_aggregated[f'global_avg_completion_time'], 5)
    global_aggregated[f'global_avg_target_id_runtime_pct'] = round(global_aggregated[f'global_avg_Target_ID_time'] / global_aggregated[f'global_avg_completion_time'], 5)
    global_aggregated[f'global_avg_neus_runtime_pct'] = round(global_aggregated[f'global_avg_NeuS_time'] / global_aggregated[f'global_avg_completion_time'], 5)

    global_aggregated[f'global_avg_cropping_video_time'] = round(global_aggregated[f'global_avg_cropping_video_time'] / global_aggregated[f'global_avg_completion_time'], 5)
    
    global_aggregated[f'global_avg_model_checks_time_pct_out_neus'] = round(global_aggregated[f'global_avg_model_checks_time'] / global_aggregated[f'global_avg_NeuS_time'], 5)
    global_aggregated[f'global_avg_model_checks_time_pct_out_neus'] = round(global_aggregated[f'global_avg_per_frame_window_detection_time'] * global_aggregated[f'global_avg_num_frame_windows'] / global_aggregated[f'global_avg_NeuS_time'], 5)
    global_aggregated[f'global_avg_model_checks_time_pct_out_neus'] = round(global_aggregated[f'global_avg_cropping_video_time'] / global_aggregated[f'global_avg_NeuS_time'], 5)
    
    global_aggregated[f'global_avg_pct_foi_out_of_full'] = round(global_aggregated[f'global_avg_pct_foi_out_of_full'], 5)

    return {"Categorical Metrics" : category_aggregated, "Global Metrics" :global_aggregated}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nsvs_result_dir")
    parser.add_argument("benchmark_name", type=str, default="lvb")
    parser.add_argument("--naive", action="store_true")
    parser.add_argument("--new_loc", action="store_true")
    args = parser.parse_args()
    nsvs_result_dir = args.nsvs_result_dir
    out_file = f'{nsvs_result_dir}/run_time_metrics.json'

    raw_res = []
    p = Path(nsvs_result_dir)
    if args.new_loc:
        found_files = p.rglob(f'**/full_time_metrics.json')
        for file_path in found_files:
            with open(file_path, "r") as file:
                out = json.load(file)
                raw_res.extend(out)
    else:
        found_files = p.rglob(f'**/postprocess_output_*.json')
        for file_path in found_files:
            with open(file_path, "r") as file:
                out = json.load(file)
                raw_res.extend(out)
    
    metrics = aggregate_metrics(raw_res, args.naive)
    
    print(metrics)

    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=4)
    get_dataframe(raw_res, args.benchmark_name, nsvs_result_dir)
    print(f'Aggregated Result saved at {out_file}')
if __name__ == "__main__":
    main()