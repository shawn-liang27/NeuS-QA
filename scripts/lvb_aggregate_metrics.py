BENCHMARK_CONFIG = { 

        "metrics":[
            "completion_time", 
            "PULS_time", 
            "Target_ID_time", 
            "NeuS_time", 
            "frame_count", 
            "num_propositions", 
            "num_frame_windows", 
            "num_model_checks", 
            "num_vlm_detections", 
            "video_IO_time", 
            "automaton_set_up_time", 
            "foi_count", 
            "pct_foi_out_of_full",
            "expected_uniform_count",
            "initial_num_frames_sampled",
            "stage1_num_frames_retained",
            "pct_stage1_from_uniform",
            "stage2_num_frames_retained",
            "pct_stage2_from_stage1",
            "pct_stage2_from_uniform",
            "final_num_frames_sampled",
            "pct_final_num_from_uniform",
            "automaton_set_up_time",
            "num_frame_windows"
            ],
        "categories": ["15", "60", "600", "3600"],
        "metadata": [
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

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd

def get_dataframe(raw_results, out_file):
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
        for k, v in result["metadata"].items():
            result[k] = v
        del result["metadata"]
    df = pd.DataFrame(raw_results)
    df.to_csv(out_file, index=False)

def get_full_report(df, metrics_list):
    """
    Computes both categorical and global metrics in one pass.
    """
    # 1. Category-specific Aggregation
    group_col = 'metadata.duration_group'
    # Calculate means for all categories at once
    cat_stats = df.groupby(group_col)[metrics_list].mean()
    
    # 2. Global Aggregation
    global_stats = df[metrics_list].mean().to_frame().T
    global_stats.index = ['Global_All']
    
    # Combine them into one master table
    combined = pd.concat([cat_stats, global_stats])
    
    # 3. Apply your Derived Metrics (Ratios and Percentages)
    # This applies the math to every row (15, 60, 600, 3600, and Global) simultaneously
    combined['PULS_runtime_pct'] = (combined['PULS_time'] / combined['completion_time']).round(3)
    combined['NeuS_runtime_pct'] = (combined['NeuS_time'] / combined['completion_time']).round(3)
    
    # Breakdown of the NeuS stage specifically
    combined['model_checks_pct_of_neus'] = (combined['model_checks_time'] / combined['NeuS_time']).round(3)
    combined['cropping_pct_of_neus'] = (combined['cropping_video_time'] / combined['NeuS_time']).round(3)

    # 4. Final Formatting
    # Ensure counts are integers
    if 'foi_count' in combined.columns:
        combined['foi_count'] = combined['foi_count'].astype(int)
        
    return combined.to_dict(orient='index')

def aggregate_benchmark(df, benchmark_config):
    
    results = {}
    
    for benchmark, config in benchmark_config.items():
        metrics = config["metrics"]
        group_col = 'metadata.duration_group'
        
        # 1. Base Aggregation: Mean of all raw metrics
        available_metrics = [m for m in metrics if m in df.columns]
        
        # Group by category and also calculate a global 'All' category
        cat_df = df.groupby(group_col)[available_metrics].mean()
        global_series = df[available_metrics].mean().to_frame().T
        global_series.index = ['Global']
        
        # Combine them so we can apply the logic to all rows at once
        combined_stats = pd.concat([cat_df, global_series])

        # 2. Derived Metrics Logic (The calculations you requested)
        # Runtime percentages relative to Completion Time
        combined_stats['avg_PULS_runtime_pct'] = (combined_stats['PULS_time'] / combined_stats['completion_time']).round(3)
        combined_stats['avg_target_id_runtime_pct'] = (combined_stats['Target_ID_time'] / combined_stats['completion_time']).round(3)
        combined_stats['avg_neus_runtime_pct'] = (combined_stats['NeuS_time'] / combined_stats['completion_time']).round(3)
        
        # Percentage of NeuS time spent on specific sub-tasks
        # (Assuming model_checks_time is a numeric sum or average)
        if 'model_checks_time' in combined_stats:
             combined_stats['avg_model_checks_pct_of_neus'] = (combined_stats['model_checks_time'] / combined_stats['NeuS_time']).round(3)

        # 3. Clean up and Formatting
        # Rename columns to match your 'avg_' naming convention if desired
        combined_stats = combined_stats.rename(columns={m: f'avg_{m}' for m in available_metrics})
        
        # Ensure integer types for counts
        if 'avg_foi_count' in combined_stats:
            combined_stats['avg_foi_count'] = combined_stats['avg_foi_count'].fillna(0).astype(int)

        # 4. Convert to final JSON structure
        results = combined_stats.to_dict(orient='index')
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nsvs_result_dir")
    parser.add_argument("--naive", action="store_true")
    args = parser.parse_args()
    nsvs_result_dir = args.nsvs_result_dir
    out_file = f'{nsvs_result_dir}/run_time_metrics.json'
    out_csv = f'{nsvs_result_dir}/metric_data.csv'
    raw_res = []
    p = Path(nsvs_result_dir)

    full_nsvs_list = []
    found_nsvs_files = p.rglob(f'**/postprocess_output_*.json')
    for file_path in found_nsvs_files:
        with open(file_path, "r") as file:
            out = json.load(file)
            for entry in out:
                full_nsvs_list.append(entry)

    found_files = p.rglob(f'**/full_time_metrics.json')
    for file_path in found_files:
        with open(file_path, "r") as file:
            out = json.load(file)
            for metrics_entry in out:
                id = metrics_entry["metadata"]["id"]
                for nsvs_entry in full_nsvs_list:
                    if id == nsvs_entry["metadata"]["id"]:
                        metrics_entry["frames_of_interest"] = nsvs_entry["frames_of_interest"]
                        metrics_entry["video_path"] = nsvs_entry["paths"]["video_path"]
                        metrics_entry["candidates"] = nsvs_entry["candidates"]
                        metrics_entry["propositions"] = nsvs_entry["puls"]["proposition"]
                        metrics_entry["question"] = nsvs_entry["candidates"]
            raw_res.extend(out)

    get_dataframe(raw_res, out_csv)

    # df = pd.read_csv(out_csv)
    # aggregated_rsults = aggregate_benchmark(df, BENCHMARK_CONFIG)
    # with open(out_file, "w") as f:
    #     json.dump(aggregated_rsults, f, indent=4)
    
    print(f'metrics csv saved at {out_csv}')
if __name__ == "__main__":
    main()