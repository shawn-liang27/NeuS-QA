from datasets import load_dataset
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir")
    args = parser.parse_args()
    dataset = load_dataset(
        "lmms-lab/Video-MME", 
        split="test"
    )

    data_list = []
    for data in dataset:
        data_list.append(data)

    with open(f"{args.data_dir}/dataset.json", "w") as f:
        json.dump(data_list, f, indent=4)
