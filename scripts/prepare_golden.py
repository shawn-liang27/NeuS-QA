import json
import random

def filter_golden_questions(results_path, val_path, output_path):
    # 1. Identify golden IDs grouped by duration
    golden_ids_by_duration = {15: [], 60: [], 600: [], 3600: []}
    
    with open(results_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            lvb = data.get("lvb_acc", {})
            
            # Check if it's a "Golden" answer
            if lvb.get("answer") == lvb.get("parsed_pred"):
                dur = lvb.get("duration_group")
                if dur in golden_ids_by_duration:
                    golden_ids_by_duration[dur].append(lvb.get("id"))

    # 2. Select the specific counts requested
    selected_ids = set()
    targets = {15: 15, 60: 15, 600: 30, 3600: 30}
    
    for dur, count in targets.items():
        available = golden_ids_by_duration[dur]
        if len(available) < count:
            print(f"Warning: Only found {len(available)} golden questions for duration {dur}, but wanted {count}.")
            selected_ids.update(available)
        else:
            selected_ids.update(random.sample(available, count))

    # 3. Filter lvb_val.json based on selected IDs
    with open(val_path, 'r') as f:
        val_data = json.load(f)

    # Assuming lvb_val.json is a list of dicts with an 'id' field
    modified_val = [q for q in val_data if q.get("id") in selected_ids]

    # 4. Save result
    with open(output_path, 'w') as f:
        json.dump(modified_val, f, indent=4)
    
    print(f"Success! Saved {len(modified_val)} golden questions to {output_path}")

# Run the function
filter_golden_questions('/usr/homes/sgl57/NeuS-VLM/NeuS-QA/experiment_results/lmm_eval/longvideobench/naive/experiment_1/InternVL2_5-8B/neusqa/OpenGVLab__InternVL2_5-8B/20260210_071820_samples_longvideobench_val_v.jsonl', '/usr/homes/sgl57/.data/LongVideoBench/lvb_val.json', '/usr/homes/sgl57/NeuS-VLM/NeuS-QA/experiment_results/nsvs_improved/ablation/adaptive_gt.json')