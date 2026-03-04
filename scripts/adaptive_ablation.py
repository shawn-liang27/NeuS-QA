from nsvqa.nsvs.video.read_video_adaptive import read_video_adaptive
import json
import time
from pathlib import Path
import ast

from sentence_transformers import SentenceTransformer
import numpy as np
import numpy as np

import numpy as np

def calculate_duration_weighted_recall(gt_indices, sampled_indices, fps=30, tolerance_sec=1.5):
    """
    Calculates Duration-Weighted Event Recall.
    
    Definition:
    1. Ground Truth indices are grouped into 'Events' (continuous segments).
    2. An Event is 'Hit' if ANY sampled frame falls within [start - tol, end + tol].
    3. The final score is the sum of durations of HIT events divided by total GT duration.
    
    Args:
        gt_indices (list): List of frame indices where GT is present.
        sampled_indices (list): List of frame indices your sampler kept.
        fps (int): Frames per second (used for grouping and tolerance).
        tolerance_sec (float): How close a sample needs to be to count as a hit (seconds).
    """
    if not gt_indices:
        return 1.0 # No events to find, so we "found" everything (trivial success)
    
    # 1. Cluster GT indices into Events
    gt_indices = sorted(list(set(gt_indices)))
    events = []
    
    if len(gt_indices) > 0:
        start = gt_indices[0]
        prev = gt_indices[0]
        
        # We group frames as one event if they are within 1 second of each other
        grouping_threshold = fps 
        
        for idx in gt_indices[1:]:
            if idx - prev > grouping_threshold:
                events.append((start, prev))
                start = idx
            prev = idx
        events.append((start, prev)) # Append the last event

    # 2. Check Hits and Calculate Weighted Recall
    total_gt_duration = 0
    detected_duration = 0
    
    # Convert tolerance to frames
    tol_frames = int(tolerance_sec * fps)
    
    for (start, end) in events:
        # Calculate duration of this specific event
        event_length = (end - start) + 1
        total_gt_duration += event_length
        
        # Check if ANY sampled frame hits this event (Binary Hit)
        # We expand the event window by the tolerance
        hit_window_start = start - tol_frames
        hit_window_end = end + tol_frames
        
        # Efficient check: is there any sample in the window?
        is_hit = any(hit_window_start <= s <= hit_window_end for s in sampled_indices)
        
        if is_hit:
            detected_duration += event_length
            
    if total_gt_duration == 0:
        return 0.0
        
    return detected_duration / total_gt_duration

def calculate_event_weighted_recall(gt_indices, sampled_indices, fps=30, iou_threshold=0.5):
    """
    Calculates recall based on 'Event Detection' rather than 'Frame Coverage'.
    
    - Groups consecutive GT frames into 'Events'.
    - If a sample lands ANYWHERE in an event (with tolerance), 
      we get credit for the FULL duration of that event.
    - Heavily penalizes missing long events; lightly penalizes missing short ones.
    """
    if not gt_indices:
        return 1.0
        
    # 1. Cluster GT indices into Events
    gt_indices = sorted(list(set(gt_indices)))

    events = []
    if gt_indices:
        start = gt_indices[0]
        prev = gt_indices[0]
        for idx in gt_indices[1:]:
            if idx - prev > fps: # Break if gap > 1 sec
                events.append((start, prev))
                start = idx
            prev = idx
        events.append((start, prev))

    # 2. Check each event
    detected_events = 0
    total_events = len(events)
    
    # Radius of influence for a single sample (e.g., +/- 1.5 seconds)
    # This means 1 sample covers ~3 seconds of content
    radius = int(fps * 1.5) 
    
    for (e_start, e_end) in events:
        event_len = (e_end - e_start) + 1
        
        # Create a boolean mask just for this event's duration
        # We normalize indices to 0..event_len for efficiency
        event_mask = np.zeros(event_len, dtype=bool)
        
        # Find samples relevant to this event
        # (Samples inside the event OR within 'radius' distance of it)
        relevant_samples = [
            s for s in sampled_indices 
            if (e_start - radius) <= s <= (e_end + radius)
        ]
        
        # Project samples onto the event mask
        for s in relevant_samples:
            # Relative position of sample to event start
            rel_s = s - e_start
            
            # Define coverage window in relative coordinates
            w_start = max(0, rel_s - radius)
            w_end = min(event_len, rel_s + radius + 1)
            
            if w_start < w_end:
                event_mask[w_start:w_end] = True
                
        # Calculate Intersection over Union (Coverage Ratio)
        covered_duration = event_mask.sum()
        coverage_ratio = covered_duration / event_len
        
        # CRITICAL: The event is only a "Hit" if we cover enough of it
        if coverage_ratio >= iou_threshold:
            detected_events += 1
            
    return detected_events / total_events

def main():
    device=5
    clip_model = SentenceTransformer('clip-ViT-B-32', device=f'cuda:{device}')
    clip_model.eval()

    with open("/usr/homes/sgl57/NeuS-VLM/NeuS-QA/experiment_results/nsvs_improved/ablation/ablation_adaptive_gt_1/gt_frames.json", "r") as f:
        data = json.load(f)

    print(f'total length {len(data)}')
    final_res = []
    for test in data:
        res = {}
        video_path = test["video_path"]
        propositions = test["propositions"]
        ground_truth_indices = test["gt_frames"]
        total_frames = test["frame_count"]
        duration = test["duration"]
        fps = round(test["fps"])
        print(propositions)
        print(type(propositions), type(propositions[0]))

        video_data = read_video_adaptive(model=clip_model, video_path=video_path, propositions=propositions, threshold=0.21)

        clip_sampled_indices = video_data["original_indices"]
        stage1_indices = video_data.get("stage1_indices", video_data["original_indices"])
        expected_unform_count = video_data["expected_uniform_count"]
        # 3. Compute Recall
        stage1_hits = 0
        hits = 0
        for gt_idx in ground_truth_indices:
            # Check if ANY sampled frame is close to this GT frame
            # Assuming indices are frame numbers and fps=30, tolerance=1sec=30frames
            if any(abs(s_idx - gt_idx) <= fps * 1.5 for s_idx in stage1_indices):
                stage1_hits += 1
            if any(abs(s_idx - gt_idx) <= fps * 1.5 for s_idx in clip_sampled_indices):
                hits += 1

        stage1_recall = stage1_hits / len(ground_truth_indices)
        adaptive_recall = hits / len(ground_truth_indices)
        stage1_weighted_recall = calculate_event_weighted_recall(gt_indices=ground_truth_indices, sampled_indices=stage1_indices, fps=fps)
        stage1_event_recall = calculate_duration_weighted_recall(gt_indices=ground_truth_indices, sampled_indices=stage1_indices, fps=fps)
        adaptive_weighted_recall = calculate_event_weighted_recall(gt_indices=ground_truth_indices, sampled_indices=clip_sampled_indices, fps=fps)
        frames_kept_pct = len(clip_sampled_indices) / total_frames
        adaptive_saved_frames_pct = len(clip_sampled_indices) / expected_unform_count
        stage1_saved_frames_pct = len(stage1_indices) / expected_unform_count

        res = {
            "video_id" : test["video_id"],
            "stage1_recall" : stage1_recall,
            "adaptive_recall" : adaptive_recall,
            "stage1_weighted_recall" : stage1_weighted_recall,
            "stage1_event_recall" : stage1_event_recall,
            "adaptive_weighted_recall" : adaptive_weighted_recall,
            "frames_kept_pct" : frames_kept_pct,
            "adaptive_frames_of_uniform_pct" : adaptive_saved_frames_pct,
            "stage1_frames_of_uniform_pct" : stage1_saved_frames_pct,
            "total_frames": total_frames,
            "duration": duration,
            "duration_group": test["duration_group"]
        }
        print(f'[DEBUG] Video: {test["id"]}\nRecall: {adaptive_recall} Weighted Recall: {adaptive_weighted_recall}\nStage1 Recall: {stage1_recall} Stage1 Weighted Recall: {stage1_weighted_recall} Stage1 Event Recall: {stage1_event_recall}\nPct of frames out of uniform: {adaptive_saved_frames_pct}')
        final_res.append(res)
    print(final_res)

    with open("/usr/homes/sgl57/NeuS-VLM/NeuS-QA/experiment_results/nsvs_improved/ablation/ablation_adaptive_gt_1/gt_analysis_result_5.json", "w") as f:
        json.dump(final_res, f, indent=2)

if __name__ == "__main__":
    main()