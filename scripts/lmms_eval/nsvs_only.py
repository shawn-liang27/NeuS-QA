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
from nsvqa.nsvs.vlm.internvl import InternVL

import json
import os
import datetime
import argparse
import logging


def exec_puls(entry, save_dir): # Step 1
    output = PULS(entry["question"], entry["metadata"]["id"], save_dir=save_dir)
    print("PULS is called")
    entry["puls"] = {}
    entry["puls"]["proposition"] = output["proposition"]
    entry["puls"]["specification"] = output["specification"]
    entry["puls"]["conversation_history"] = os.path.join(os.getcwd(), output["saved_path"])

def exec_target_identification(entry, save_dir): # Step 2
    output = identify_target(
        entry["question"],
        entry["candidates"],
        entry["puls"]["specification"],
        entry["puls"]["conversation_history"],
        entry["metadata"]["id"],
        save_dir
    )

    entry["target_identification"] = {}
    entry["target_identification"]["frame_window"] = output["frame_window"]
    entry["target_identification"]["explanation"] = output["explanation"]
    entry["target_identification"]["conversation_history"] = os.path.join(os.getcwd(), output["saved_path"])

def exec_nsvs(entry, sample_rate, device, model, vlm): # Step 3
    print(entry["paths"]["video_path"])
    reader = Mp4Reader(path=entry["paths"]["video_path"], sample_rate=sample_rate)
    video_data = reader.read_video()
    if "metadata" not in entry:
        entry["metadata"] = {}
    entry["metadata"]["fps"] = video_data["video_info"]["fps"]
    entry["metadata"]["frame_count"] = video_data["video_info"]["frame_count"]
    try:
        output, indices = run_nsvs(
            video_data,
            entry["paths"]["video_path"],
            entry["puls"]["proposition"],
            entry["puls"]["specification"],
            device=device,
            model=model,
            vlm=vlm,
        )
    except Exception as e:
        entry["metadata"]["error"] = repr(e)
        output = [-1]
        indices = []
    
    entry["nsvs"] = {}
    entry["nsvs"]["output"] = output
    entry["nsvs"]["indices"] = indices

def exec_merge(entry): # Step 4
    inner = entry["target_identification"]["frame_window"].strip()[1:-1]
    parts = inner.split(',')
    result = []
    for part in parts:
        part = part.strip()
        match = re.search(r'([+-])\s*(\d+)', part)
        if match:
            sign, num = match.groups()
            result.append(int(sign + num))
        else:
            result.append(0)

    if entry["nsvs"]["output"] != [-1]:
        entry["frames_of_interest"] = [
            max(0,                                  int(entry["nsvs"]["output"][0] + result[0] * entry["metadata"]["fps"])),
            min(entry["metadata"]["frame_count"]-1, int(entry["nsvs"]["output"][1] + result[1] * entry["metadata"]["fps"]))
        ]
    else:
        entry["frames_of_interest"] = [-1]

def run_nsvqa(output_dir, llm_convo_dir, current_split, total_splits, vlm_config, data_dir, data_loader):
    data = data_loader.load_data()
    print(f'Data Loading Complete! Data Length {len(data)}\nStarting NSVS Module')
    output = []
    starting = (len(data) * (current_split-1)) // total_splits
    ending = (len(data) * current_split) // total_splits
    vlm = InternVL(model_name=vlm_config[1], device=vlm_config[0])
    for i in range(starting, ending):
        print("\n" + "*"*50 + f" {i}/{len(data)-1} " + "*"*50)
        entry = data[i]
        exec_puls(entry, llm_convo_dir)
        exec_target_identification(entry, llm_convo_dir)
        exec_nsvs(entry, sample_rate=1, device=vlm_config[0], model=vlm_config[1], vlm=vlm)
        exec_merge(entry)
        output.append(entry)

    with open(output_dir, "w") as f:
        json.dump(output, f, indent=4)

def main(args):
    vlm_config = (args.port_number, args.vlm_model_name) # device_number, model_name

    experiment_dir = args.output_dir
    current_split = args.current_split

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(f'{experiment_dir}/nsvqa_output', exist_ok=True)
    os.makedirs(f'{experiment_dir}/vqa_output', exist_ok=True)
    os.makedirs(f'{experiment_dir}/postprocess_output', exist_ok=True)

    nsvqa_dir = f"{experiment_dir}/nsvqa_output/nsvqa_output_{current_split}.json"
    vqa_dir = f"{experiment_dir}/vqa_output/vqa_output_{current_split}.json"
    postprocess_dir = f"{experiment_dir}/postprocess_output/postprocess_output_{current_split}.json"
    nsvs_llm_convo_dir = f"{experiment_dir}/llm_conversation_history/"

    print(f'Loading Data from Data_Dir: {args.data_dir}\nBurned_Dir: {args.burned_dir}')

    data_loader = LongVideoBench(dataset_path=args.data_dir, burned_path=args.burned_dir, postprocess_dir=postprocess_dir, categories=args.categories)

    run_nsvqa(output_dir=nsvqa_dir, llm_convo_dir=nsvs_llm_convo_dir, current_split=args.current_split, total_splits=args.total_splits, vlm_config=vlm_config, data_dir=args.data_dir, data_loader=data_loader)

    data_loader.postprocess_data(nsvqa_dir)
    # if args.use_lmm_evals:
    #     lmm_eval_vqa(postprocess_dir, vqa_dir, vlm_config, max_num_frames=args.max_num_frames, eval=True, pure_vqa=args.pure_vqa)
    # else:
    #     vqa(postprocess_dir, vqa_dir, vlm_config, max_num_frames=args.max_num_frames, eval=True)

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
    # parser.add_argument("--use_lmm_evals", action='store_true', default = True)
    # parser.add_argument("--pure_vqa", action='store_true', default = False)
    # parser.add_argument("--max_num_frames", type=int, default = 32)

    
    args = parser.parse_args()
    
    # experiment_dir = args.output_dir
    # os.makedirs(experiment_dir, exist_ok=True)
    # os.makedirs(f'{experiment_dir}/nsvqa_output', exist_ok=True)
    # os.makedirs(f'{experiment_dir}/vqa_output', exist_ok=True)
    # os.makedirs(f'{experiment_dir}/postprocess_output', exist_ok=True)

    # vlm_config = (args.port_number, args.vlm_model_name) # device_number, model_name

    # main(experiment_dir, vlm_config, args.data_dir, args.burned_dir, args.categories, args.current_split, args.total_splits)
    
    main(args)

