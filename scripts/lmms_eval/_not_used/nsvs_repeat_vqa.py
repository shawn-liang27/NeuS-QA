from nsvqa.vqa.vqa import vqa
from nsvqa.vqa.lmm_vqa import lmm_eval_vqa

import json
import os
import datetime
import argparse
import logging

def main(args):
    vlm_config = (args.port_number, args.vlm_model_name)
    experiment_dir = args.experiment_dir

    current_datetime = datetime.datetime.now()
    current_split = args.current_split
    os.makedirs(args.out_dir, exist_ok=True)

    vqa_output = f"{args.out_dir}/vqa_output_{current_split}.json"
    postprocess_dir = f"{experiment_dir}/postprocess_output/postprocess_output_{current_split}.json"
    lmm_eval_vqa(postprocess_dir, vqa_output, vlm_config, max_num_frames=args.max_num_frames, eval=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir")
    parser.add_argument("--out_dir")
    parser.add_argument("--vlm_model_name", type=str)
    parser.add_argument("--port_number", type=int)
    parser.add_argument("--max_num_frames", type=int)
    parser.add_argument("--current_split", type=int)
    args = parser.parse_args()
    main(args)
