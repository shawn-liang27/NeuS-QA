from pathlib import Path
import json
import argparse
from collections import defaultdict

def main(out_dir, prefix):

    out_json = f"{out_dir}/{prefix}_combined_res.json"
    
    combined_res = defaultdict(dict)
    p = Path(out_dir)
    found_files = p.rglob(f'**/vqa_summary.json')
    for file_path in found_files:
        with open(file_path, "r") as file:
            out = json.load(file)
            for category, res_dict in out.items():
                new_total = combined_res[category].get("total", 0) + res_dict["total"]

                new_num_correct = combined_res[category].get("num_correct", 0) + res_dict["num_correct"]
                
                new_accuracy = new_num_correct / new_total

                combined_res[category]["total"] = new_total
                combined_res[category]["num_correct"] = new_num_correct
                combined_res[category]["accuracy"] = new_accuracy
                    
    with open(out_json, "w") as file:
        json.dump(combined_res, file, indent=4)
    print(f"Combined results are saved in {out_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()

    main(args.out_dir, args.prefix)