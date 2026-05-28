from nsvqa.target_identification.target_identification import *
from nsvqa.nsvs.model_checker.frame_validator import *
from nsvqa.datamanager.longvideobench import *
from nsvqa.nsvs.video.read_video import *
from nsvqa.datamanager.custom import *
from nsvqa.nsvs.vlm.obj import *
from nsvqa.nsvs.nsvs import *
from nsvqa.puls.puls import *
from nsvqa.vqa.vqa import vqa
from nsvqa.vqa.lmm_vqa import lmm_eval_vqa
# from nsvqa.nsvs.vlm.internvl import InternVL
# from nsvqa.nsvs.vlm.vllm_client import VLLMClient
from nsvqa.nsvs.vlm.internvl_precompute import InternVL
from sentence_transformers import SentenceTransformer
import json
import os
import datetime
import argparse
import logging
from pathlib import Path

import time
from collections import defaultdict

from nsvqa.nsvs.video.read_video_adaptive import read_video_adaptive

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def exec_puls(entry, save_dir): # Step 1
    print("PULS is called")
    output = PULS(entry["question"], entry["metadata"]["id"], save_dir=save_dir)
    print(f'[DEBUG] PULS Output: {output}' )
    entry["puls"] = {}
    entry["puls"]["proposition"] = output["proposition"]
    entry["puls"]["specification"] = output["specification"]
    entry["puls"]["conversation_history"] = os.path.join(os.getcwd(), output["saved_path"])

def exec_target_identification(entry, save_dir): # Step 2
    print("Target ID is called")
    output = identify_target(
        entry["question"],
        entry["candidates"],
        entry["puls"]["specification"],
        entry["puls"]["conversation_history"],
        entry["metadata"]["id"],
        save_dir
    )
    print(f'[DEBUG] Target ID Output: {output}' )
    entry["target_identification"] = {}
    entry["target_identification"]["frame_window"] = output["frame_window"]
    entry["target_identification"]["explanation"] = output["explanation"]
    entry["target_identification"]["conversation_history"] = os.path.join(os.getcwd(), output["saved_path"])

def exec_nsvs(entry, sample_rate, device, model, clip_model, vlm, measure_metrics): # Step 3
    print(f'NeuS Module is Called {entry["paths"]["video_path"]}')

    # 1. Video IO time
    io_start = time.perf_counter() if measure_metrics else 0
    
    # video_data = get_relevant_frames_from_video(model=clip_model, video_path=entry["paths"]["video_path"], propositions=entry["puls"]["proposition"], threshold=0.22)

    video_data = read_video_adaptive(model=clip_model, video_path=entry["paths"]["video_path"], propositions=entry["puls"]["proposition"], threshold=0.22)
    
    if measure_metrics: 
        entry["time_metrics"]["video_IO_time"] = time.perf_counter() - io_start
        clip_metrics = video_data.get("clip_metrics", {})
        for metric, value in clip_metrics.items():
            entry["time_metrics"][metric] = value
        entry["time_metrics"]["final_num_frames_sampled"] = video_data["final_num_frames_sampled"]
        entry["time_metrics"]["video_sample_method"] = video_data["video_sample_method"]
        entry["time_metrics"]["pct_final_num_from_uniform"] = video_data["pct_final_num_from_uniform"]
    

    if "metadata" not in entry:
        entry["metadata"] = {}
    
    entry["metadata"]["fps"] = video_data["video_info"]["fps"]
    entry["metadata"]["frame_count"] = video_data["video_info"]["frame_count"]
    # entry["metadata"]["num_of_frame_in_sequence"] = num_of_frame_in_sequence
    try:
        output, indices, frames_of_interest, run_metrics = run_nsvs(
            video_data,
            entry["paths"]["video_path"],
            entry["puls"]["proposition"],
            entry["puls"]["specification"],
            entry["target_identification"]["frame_window"],
            device=device,
            model=model,
            vlm=vlm,
            measure_metrics=measure_metrics
        )
    except Exception as e:
        entry["metadata"]["error"] = repr(e)
        output = [-1]
        indices = []
        run_metrics = {}
        frames_of_interest= [-1]
        print(f"DEBUG: run_nsvs failed with: {e}")
        traceback.print_exc()
    entry["nsvs"] = {}
    entry["nsvs"]["output"] = output
    entry["nsvs"]["indices"] = [list(s) for s in indices]
    entry["frames_of_interest"] = frames_of_interest
    
    if measure_metrics:
        run_metrics["frames_of_interest"] = frames_of_interest
        for metric, value in run_metrics.items():
            entry["time_metrics"][metric] = value

def run_nsvqa(output_dir, llm_convo_dir, current_split, total_splits, vlm_config, data_dir, data_loader, measure_metrics=False):
    data = data_loader.load_data()
    print(f'Data Loading Complete! Data Length {len(data)}\nStarting NSVS Module')
    output = []
    metrics_output = []
    starting = (len(data) * (current_split-1)) // total_splits
    ending = (len(data) * current_split) // total_splits
    vlm = InternVL(model_name=vlm_config[1], device=vlm_config[0], max_patch=1)
    # vlm = VLLMClient(model=vlm_config[1], api_base=f"http://localhost:{vlm_config[0]}/v1")
    print("Loading CLIP model to GPU...")
    clip_model = SentenceTransformer('clip-ViT-B-32', device=f'cuda:{vlm_config[0]}')
    clip_model.eval()

    for i in range(starting, ending):
        print("\n" + "*"*50 + f" {i}/{len(data)-1} " + "*"*50)
        metrics = {} if measure_metrics else None
        entry = data[i]
        if measure_metrics: 
            print("time metrics added")
            entry["time_metrics"] = {}
        # 1. Start Total Completion Timer
        t_start = time.perf_counter() if measure_metrics else 0

        # 2. Module: PULS
        p_start = time.perf_counter() if measure_metrics else 0
        exec_puls(entry, llm_convo_dir)
        if measure_metrics: 
            metrics["PULS_time"] = time.perf_counter() - p_start
            metrics["num_propositions"] = len(entry["puls"]["proposition"])

        if not entry["puls"]["proposition"]:
            entry["nsvs"] = {}
            entry["nsvs"]["output"] = []
            entry["nsvs"]["indices"] = []
            if "metadata" not in entry:
                entry["metadata"] = {}
            entry["metadata"]["error"] = f"PULS failed to process Propositions and Specification for Task: {entry["metadata"].get("id", "Unknown")}, Returning Full Video Length"
            entry["frames_of_interest"] = [-1]
            logging.critical(f'PULS failed to process Propositions and Specification for Task: {entry["metadata"].get("id", "Unknown")}, Returning Full Video Length')
            continue

        # 3. Module: Target Identification
        tid_start = time.perf_counter() if measure_metrics else 0
        exec_target_identification(entry, llm_convo_dir)
        if measure_metrics: metrics["Target_ID_time"] = time.perf_counter() - tid_start

        # 4. Module: NSVS (The primary bottleneck)
        n_start = time.perf_counter() if measure_metrics else 0
        # Pass the pre-initialized vlm here
        exec_nsvs(entry, sample_rate=1, device=vlm_config[0], model=vlm_config[1], clip_model=clip_model, vlm=vlm, measure_metrics=measure_metrics)
        print(f'Neus Complete with question {entry["metadata"]["id"]}')
        if measure_metrics: metrics["NeuS_time"] = time.perf_counter() - n_start

        # 5. Finalize Total Time
        if measure_metrics:
            metrics["completion_time"] = time.perf_counter() - t_start
            metrics["metadata"] = entry.get("metadata", {}).copy()

            time_metrics = entry.pop("time_metrics", {})
            metrics.update(time_metrics)
            print(f'Runtime metrics on {entry["metadata"]["id"]}:\n{metrics}')
            metrics_output.append(metrics)

        output.append(entry)
        
        vlm.clear_gpu_memory()

    with open(output_dir, "w") as f:
        json.dump(output, f, indent=4)

    if measure_metrics:
        metric_output_path = f'{str(Path(output_dir).parent.parent)}/full_time_metrics.json'
        with open(metric_output_path, "w") as f:
            json.dump(metrics_output, f, indent=4)
    

def main(args): 
    vlm_config = (args.port_number, args.vlm_model_name) # device_number, model_name

    experiment_dir = args.output_dir
    current_split = args.current_split

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(f'{experiment_dir}/nsvqa_output', exist_ok=True)
    # os.makedirs(f'{experiment_dir}/vqa_output', exist_ok=True)
    os.makedirs(f'{experiment_dir}/postprocess_output', exist_ok=True)

    nsvqa_dir = f"{experiment_dir}/nsvqa_output/nsvqa_output_{current_split}.json"
    vqa_dir = f"{experiment_dir}/vqa_output/vqa_output_{current_split}.json"
    postprocess_dir = f"{experiment_dir}/postprocess_output/postprocess_output_{current_split}.json"
    nsvs_llm_convo_dir = f"{experiment_dir}/llm_conversation_history/"

    print(f'Loading Data from Data_Dir: {args.data_dir}\nBurned_Dir: {args.burned_dir}')
        
    data_loader = LongVideoBench(dataset_path=args.data_dir, burned_path=args.burned_dir, postprocess_dir=postprocess_dir, categories=args.categories)

    run_nsvqa(output_dir=nsvqa_dir, llm_convo_dir=nsvs_llm_convo_dir, current_split=args.current_split, total_splits=args.total_splits, vlm_config=vlm_config, data_dir=args.data_dir, data_loader=data_loader, measure_metrics=args.measure_metrics)
    # Time PostProcess
    data_loader.postprocess_data(nsvqa_dir, args.measure_metrics)

    # if args.measure_metrics:
    #     try:
    #         runtime_metrics_dir = f"{experiment_dir}/runtime_metrics"
    #         os.makedirs(runtime_metrics_dir, exist_ok=True)
    #         with open(postprocess_dir, "r") as file:
    #             raw_result = json.load(file)
    #         metrics_result = aggregate_metrics(raw_result)
    #         print(f"Saving Time Metrics Result to '{runtime_metrics_dir}/runtime_metrics_{current_split}.json'")
    #         with open(f'{runtime_metrics_dir}/runtime_metrics_{current_split}.json', "w") as f:
    #             json.dump(metrics_result, f, indent=4)
    #     except Exception as e:
    #         print()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm_model_name", type=str)
    parser.add_argument("--port_number", type=int)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--burned_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--current_split", type=int)
    parser.add_argument("--total_splits", type=int)
    parser.add_argument('--categories', nargs='+', type=str)
    parser.add_argument("--measure_metrics", action='store_true', default = False)
    
    args = parser.parse_args()
    main(args)

