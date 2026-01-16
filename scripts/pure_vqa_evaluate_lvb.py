import json
import os
import datetime
import argparse
import logging
from nsvqa.vqa.pure_vqa import *
from nsvqa.datamanager.longvideobench import *
from nsvqa.vqa.lmm_vqa import lmm_eval_vqa

# def main(experiment_dir, vlm_config, data_dir, burned_dir, categories, current_split, total_splits, max_num_frames):
def main(args):

    experiment_dir = f'{args.output_dir}/vqa_output'
    os.makedirs(experiment_dir, exist_ok=True)
    
    vlm_config = (args.port_number, args.vlm_model_name) # device_number, model_name
    current_split = args.current_split
    total_splits = args.total_splits
    vqa_dir = f"{experiment_dir}/vqa_output_{current_split}.json"
    
    data_loader = LongVideoBench(dataset_path=args.data_dir, burned_path=args.burned_dir, postprocess_dir="", categories=args.categories)
    data = data_loader.load_data()

    output = []
    starting = (len(data) * (current_split-1)) // total_splits
    ending = (len(data) * current_split) // total_splits

    data_split = []

    for i in range(starting, ending):
        data_split.append(data[i])

    print("\n" + "*"*100)
    print(f"Data Split Complete. \nTotal Length {len(data)} \nCurrent Split: {current_split}\nTotal Split: {total_splits} \nIndex Range: [{starting, ending}]")
    print("\n" + "*"*100)
    if args.use_lmm_evals:
        lmm_eval_vqa(data_split, vqa_dir, vlm_config, args.max_num_frames, pure_vqa=args.pure_vqa, eval=True)
    else:
        vqa(data_split, vqa_dir, vlm_config, args.max_num_frames, pure_vqa=args.pure_vqa, eval=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm_model_name", type=str)
    parser.add_argument("--port_number", type=int)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--burned_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--current_split", type=int)
    parser.add_argument("--total_splits", type=int)
    parser.add_argument("--use_lmm_evals", action='store_true', default = True)
    parser.add_argument("--pure_vqa", action='store_true', default = False)
    parser.add_argument('--categories', nargs='+', type=str)
    parser.add_argument("--max_num_frames", type=int, default = 32)

    args = parser.parse_args()
    
    main(args)

    # main(experiment_dir, vlm_config, args.data_dir, args.burned_dir, args.categories, args.current_split, args.total_splits, args.max_num_frames)

