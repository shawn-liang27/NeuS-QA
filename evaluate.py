from nsvqa.target_identification.target_identification import *
from nsvqa.nsvs.model_checker.frame_validator import *
from nsvqa.datamanager.longvideobench import *
from nsvqa.nsvs.video.read_video import *
from nsvqa.datamanager.custom import *
from nsvqa.nsvs.vlm.obj import *
from nsvqa.nsvs.nsvs import *
from nsvqa.puls.puls import *
from nsvqa.vqa.vqa import *

import json
import os
import datetime
import argparse

def exec_puls(entry, save_dir): # Step 1
    output = PULS(entry["question"], save_dir=save_dir)

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
        save_dir
    )

    entry["target_identification"] = {}
    entry["target_identification"]["frame_window"] = output["frame_window"]
    entry["target_identification"]["explanation"] = output["explanation"]
    entry["target_identification"]["conversation_history"] = os.path.join(os.getcwd(), output["saved_path"])

def exec_nsvs(entry, sample_rate, device, model): # Step 3
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

def run_nsvqa(output_dir, llm_convo_dir, current_split, total_splits, vlm_config, video_path):
    # loader = LongVideoBench()
    loader = Custom(
        raw_data=[
            {
                "video_path": video_path,
                "question": "What happens when wine shows up on the screen before the vineyards showed up on the screen?",
                "answer_choices": [
                    "A close up of the wine was shown",
                    "The wine was trashed",
                    "The wine was replaced with soda",
                    "The man in the blue shirt was talking"
                ]
            }
        ]
    )
    data = loader.load_data()
    
    output = []
    starting = (len(data) * (current_split-1)) // total_splits
    ending = (len(data) * current_split) // total_splits
    for i in range(starting, ending+1):
        print("\n" + "*"*50 + f" {i}/{len(data)-1} " + "*"*50)
        entry = data[i]
        exec_puls(entry, llm_convo_dir)
        exec_target_identification(entry, llm_convo_dir)
        exec_nsvs(entry, sample_rate=1, device=vlm_config[0], model=vlm_config[1])
        exec_merge(entry)
        output.append(entry)

    with open(output_dir, "w") as f:
        json.dump(output, f, indent=4)

def postprocess(nsvqa_dir, postprocess_dir):
    loader = Custom(postprocess_dir=postprocess_dir)
    loader.postprocess_data(nsvqa_dir)

def main(experiment_dir, vlm_config, example_vid_path):
    current_split = 1 # split between GPUs
    total_splits = 4

    nsvqa_dir = f"{experiment_dir}/nsvqa_output/nsvqa_output_{current_split}.json"
    vqa_dir = f"{experiment_dir}/vqa_output/vqa_output_{current_split}.json"
    postprocess_dir = f"{experiment_dir}/postprocess_output/postprocess_output_{current_split}.json"
    nsvs_llm_convo_dir = f"{experiment_dir}/llm_conversation_history/"
    
    print(nsvqa_dir, vqa_dir, postprocess_dir)

    run_nsvqa(nsvqa_dir, nsvs_llm_convo_dir, current_split, total_splits, vlm_config, example_vid_path)
    postprocess(nsvqa_dir, postprocess_dir)
    vqa(postprocess_dir, vqa_dir, vlm_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm_model_name")
    parser.add_argument("--port_number", type=int)
    parser.add_argument("--output_dir")
    parser.add_argument("--example_vid_path", help=f"burned in subtitle video: mH9LdC7IFH8.mp4")

    args = parser.parse_args()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"{args.output_dir}/nsvs_qa_{timestamp}"
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(f'{experiment_dir}/nsvqa_output', exist_ok=True)
    os.makedirs(f'{experiment_dir}/vqa_output', exist_ok=True)
    os.makedirs(f'{experiment_dir}/postprocess_output', exist_ok=True)

    vlm_config = (args.port_number, args.vlm_model_name) # device_number, model_name

    main(experiment_dir, vlm_config, args.example_vid_path)

