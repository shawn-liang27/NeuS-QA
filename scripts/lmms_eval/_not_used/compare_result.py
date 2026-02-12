import argparse
import json
from pathlib import Path
import os
from datetime import datetime

now = datetime.now()
current_datetime = now.strftime("%Y_%m_%d_%H_%M_%S")

def main(args):

    nsvs_res = []
    pure_vqa_res = []
    nsvs_res_dir = Path(args.nsvs_res_dir)
    vqa_res_dir = Path(args.vqa_res_dir)

    out_dir = Path(f'{args.out_dir}/{current_datetime}')
    out_dir.mkdir(parents=True, exist_ok=True)

    for file in nsvs_res_dir.rglob("vqa_output_[1-4].json"):
        with open(file, "r") as res_file:
            res = json.load(res_file)
            nsvs_res.extend(res)

    for file in vqa_res_dir.rglob("vqa_output_[1-4].json"):
        with open(file, "r") as res_file:
            res = json.load(res_file)
            pure_vqa_res.extend(res)
    
    comparison = []
    nsvs_wrong = []

    nsvs_better = []
    
    both_wrong_count = 0
    both_right_count = 0

    for nsvs_output in nsvs_res:
        cur = {}
        nsvs_id = nsvs_output["id"]
        
        for vqa_output in pure_vqa_res:
            vqa_id = vqa_output["id"]
            if nsvs_id == vqa_id:
                
                cur["id"] = nsvs_id
                cur["question"] = nsvs_output["question"]
                cur["candidates"] = vqa_output["candidates"]
                cur["question_category"] = vqa_output["question_category"]

                cur["nsvs_prediction"] = nsvs_output["parsed_prediction"]
                cur["vqa_prediction"] = vqa_output["parsed_prediction"]
                cur["correct_answer"] = vqa_output["parsed_prediction"]
                nsvs_correct = nsvs_output["is_correct"]
                vqa_correct = vqa_output["is_correct"]

                cur["nsvs_video_path"] = nsvs_output["video_path"]
                cur["vqa_video_path"] = vqa_output["video_path"]


                if nsvs_correct and vqa_correct:
                    cur["correctness"] = "Both Correct"
                    both_right_count += 1
                elif not nsvs_correct and not vqa_correct:
                    cur["correctness"] = "Both Wrong"
                    both_wrong_count += 1
                elif nsvs_correct:
                    cur["correctness"] = "nsvs"
                    nsvs_better.append(cur)
                else:
                    cur["correctness"] = "vqa"
                    nsvs_wrong.append(cur)
                
                comparison.append(cur)
                break
    
    with open(f"{out_dir}/complete_result_comparison.json", "w") as file:
        json.dump(comparison, file, indent=4)
    
    with open(f"{out_dir}/nsvs_worse.json", "w") as file:
        json.dump(nsvs_wrong, file, indent=4)

    with open(f"{out_dir}/nsvs_better.json", "w") as file:
        json.dump(nsvs_better, file, indent=4)

    metadata = {}
    metadata["nsvs_result_dir"] = str(nsvs_res_dir)
    metadata["vqa_result_dir"] = str(vqa_res_dir)

    metadata["nsvs_model"] = nsvs_res_dir.parent.name
    metadata["vqa_model"] = vqa_res_dir.parent.name

    metadata["total_questions_compared"] = len(comparison)

    total_by_cat = {}

    for case in comparison:
            total_by_cat[case["question_category"]] = total_by_cat.get(case["question_category"], 0) + 1

    metadata["total_questions_compared_by_category"] = total_by_cat
    metadata["total_nsvs_worse"] = len(nsvs_wrong)
    metadata["total_nsvs_better"] = len(nsvs_better)

    nsvs_worse_by_category = {}
    for nsvs_worse_case in nsvs_wrong:
        nsvs_worse_by_category[nsvs_worse_case["question_category"]] = nsvs_worse_by_category.get(nsvs_worse_case["question_category"], 0) + 1
    metadata["total_nsvs_worse_by_cat"] = nsvs_worse_by_category
    
    nsvs_better_by_category = {}
    for nsvs_better_case in nsvs_better:
        nsvs_better_by_category[nsvs_better_case["question_category"]] = nsvs_better_by_category.get(nsvs_better_case["question_category"], 0) + 1

    metadata["total_nsvs_better_by_cat"] = nsvs_better_by_category

    with open(f"{out_dir}/comparison_metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsvs_res_dir", type=str)
    parser.add_argument("--vqa_res_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    args = parser.parse_args()
    main(args)


