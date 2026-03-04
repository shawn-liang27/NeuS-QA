from enum import auto
import numpy as np
import warnings
import tqdm
import os
import traceback

from nsvqa.nsvs.model_checker.property_checker import PropertyChecker
from nsvqa.nsvs.model_checker.video_automaton import VideoAutomaton
from nsvqa.nsvs.video.frames_of_interest import FramesofInterest
from nsvqa.utils.intersection import intersection_with_gaps, reconcile_sparse_ltl, group_with_gaps, resolve_target_window
from nsvqa.nsvs.video.video_frame import VideoFrame
from nsvqa.nsvs.vlm.vllm_client import VLLMClient

import logging
import time
import sys

PRINT_ALL = False

BRIDGE_MULT =  10 # seconds
CONTEXT_SECONDS = 10 # Look for P/Q within 10 seconds of a handover

warnings.filterwarnings("ignore")

def group_into_continuous_events(frames_with_indices, gap_threshold=500):
    """
    Groups (frame, index) tuples into 'event clusters' based on time gaps.
    """
    if not frames_with_indices:
        return []

    events = []
    current_event = [frames_with_indices[0]]

    for i in range(1, len(frames_with_indices)):
        prev_idx = frames_with_indices[i-1][1]
        curr_idx = frames_with_indices[i][1]

        # If the gap is too big, start a new event cluster
        if curr_idx - prev_idx > gap_threshold:
            events.append(current_event)
            current_event = []
        
        current_event.append(frames_with_indices[i])
    
    events.append(current_event) # Add the last one
    return events

def run_nsvs_ablation(
    video_data: dict,
    video_path: str,
    proposition: list,
    specification: str,
    target_window: str,
    model: str,
    device: int,
    vlm: str,
    measure_metrics: bool,
    config: dict,
    model_type: str = "dtmc",
    num_of_frame_in_sequence = 3,
    tl_satisfaction_threshold: float = 0.6,
    detection_threshold: float = 0.5,
    vlm_detection_threshold: float = 0.349,
    image_output_dir: str = "output",
):
    """
    Find relevant frames from a video that satisfy a specification
    Args:
        video_data:
            {
                "images": list[numpy.array],
                "original_indices": list[int],
                "video_info": 
                    {
                        "sample_rate": sample_rate
                        "fps": fps,
                        "frame_count": frame_count
                    }
            }
          
    """
    def process_frame_batching(sequence_of_frames: list[np.ndarray], current_indices: int, measure_metrics: bool, proposition: list):
        object_of_interest = {}    

        detected_objects = vlm.batch_detect(
            seq_of_frames=sequence_of_frames,
            scene_descriptions=proposition,
            threshold=vlm_detection_threshold
        )

        for detected_object in detected_objects:
            object_of_interest[detected_object.name] = detected_object      

        frame = VideoFrame(
            frame_idx_list=current_indices,
            object_of_interest=object_of_interest,
        )
        return frame

    def process_frame_sequential(sequence_of_frames: list[np.ndarray], current_indices: int, measure_metrics: bool, proposition: list):
        object_of_interest = {}    
        for prop in proposition:
            detected_object = vlm.detect(
                seq_of_frames=sequence_of_frames,
                scene_description=prop,
                threshold=vlm_detection_threshold
            )
            object_of_interest[prop] = detected_object

        frame = VideoFrame(
            frame_idx_list=current_indices,
            object_of_interest=object_of_interest,
        )
        return frame
    
    process_frame_method = process_frame_sequential if config.vlm_method == "sequential" else process_frame_batching

    if PRINT_ALL:
        print(f"Propositions: {proposition}\n")
        print(f"Specification: {specification}\n")
        print(f"Video path: {video_path}\n")
    
    if measure_metrics:
        time_metrics = {}

    def log_metrics(target_key, value, is_list=False):
        if is_list:
            time_metrics.setdefault(target_key, []).append(value)
        else:
            time_metrics[target_key] = value

    _model_check_count = 0
    _vlm_detection_count = 0

    # Time automaton set up
    setup_start = time.perf_counter() if measure_metrics else 0
    automaton = VideoAutomaton(include_initial_state=True)
    automaton.set_up(proposition_set=proposition)
    
    checker = PropertyChecker(
        proposition=proposition,
        specification=specification,
        model_type=model_type,
        tl_satisfaction_threshold=tl_satisfaction_threshold,
        detection_threshold=detection_threshold
    )

    if measure_metrics: 
        setup_duration = time.perf_counter() - setup_start
        log_metrics("automaton_set_up_time", setup_duration)

    frames_with_indices = [
        (img, idx) for img, idx in zip(video_data["images"], video_data["original_indices"])
    ]
    frame_of_interest = FramesofInterest()
    # 1. Get continuous clusters from your CLIP output
    event_clusters = group_into_continuous_events(frames_with_indices, gap_threshold=240)

    frame_windows = []

    for event in event_clusters:
        # Only create windows if the event has enough frames
        # Sliding window logic within a single continuous event
        for i in range(0, len(event), num_of_frame_in_sequence):
            window = event[i : i + num_of_frame_in_sequence]
            
            # Optional: If a window is too small (e.g., only 1 frame), 
            # you might choose to skip it or pad it.
            if len(window) > 0:
                frame_windows.append(window)
    print(f'[DEBUG] Num frame_window: {len(frame_windows)} {len(frame_windows[0])}')
    if PRINT_ALL:
        looper = enumerate(frame_windows)
    else:
        looper = tqdm.tqdm(enumerate(frame_windows), total=len(frame_windows))

    if measure_metrics:
        log_metrics("num_frame_windows", len(frame_windows))

    all_detections = [set(), set()]
    gt_indices = set()
    for i, window in looper:
        if PRINT_ALL:
            print("\n" + "*"*50 + f" {i}/{len(frame_windows)-1} " + "*"*50)
            print("Detections:")

        current_frames = [f[0] for f in window]
    
        # 2. Extract the indices (index 1) for your automaton/logic
        current_indices = [f[1] for f in window]

        per_window_detection_start = time.perf_counter() if measure_metrics else 0

        try:
            frame = process_frame_method(current_frames, current_indices, measure_metrics, proposition)
        except Exception as e:
            print(f"[FATAL] Caught Exception: {e}\nExiting the Program...")
            traceback.print_exc()
            sys.exit(1) 
        if measure_metrics: 
            per_window_detection_duration = time.perf_counter() - per_window_detection_start
            log_metrics("per_frame_window_detection_time", per_window_detection_duration, True)

        # if PRINT_ALL:
        #     print("Detections Completed and Returned")
        # if PRINT_ALL and False: # disabled
        #     os.makedirs(image_output_dir, exist_ok=True)
        #     frame.save_frame_img(save_path=os.path.join(image_output_dir, f"{i}"))
        if config.adaptive_gt:
            high_conf_objects = frame.thresholded_detected_objects(0.8)
            if len(high_conf_objects) > 0:
                print(f"GT Frame detected: idx {frame.frame_idx_list}")
                # Add all indices in this batch (since they share the detection)
                gt_indices.update(frame.frame_idx_list)

        if checker.validate_frame(frame_of_interest=frame):
            thresh = frame.thresholded_detected_objects(threshold=detection_threshold)
            for prop in thresh.keys():
                split = checker.check_split(prop)
                all_detections[split].update(frame.frame_idx_list)
            if PRINT_ALL:
                print(f"\t{all_detections}")

            add_automaton_model_check_start = time.perf_counter() if measure_metrics else 0
            automaton.add_frame(frame=frame)
            frame_of_interest.add_frame(frame)

            model_check = checker.check_automaton(automaton=automaton)

            _model_check_count += 1

            if model_check:
                automaton.reset()
                frame_of_interest.flush_frame_buffer()

            if measure_metrics: 
                per_model_check_duration = time.perf_counter() - add_automaton_model_check_start
                log_metrics("model_checks_time", per_model_check_duration, True)

    automaton_foi = frame_of_interest.compile_foi()

    if config.return_segments:
        if PRINT_ALL:
            print(f"Automaton indices (Actual): {automaton_foi}")

        foi = reconcile_sparse_ltl(
            video_info=video_data['video_info'],
            p_indices=all_detections[0], 
            q_indices=all_detections[1],
            automaton_foi=automaton_foi,
            target_window=target_window,
            BRIDGE_MULT=BRIDGE_MULT,
            CONTEXT_SECONDS=CONTEXT_SECONDS,
            TOLERANCE=8
        )

        if not foi or foi == [-1]:
            MAX_GAPS_CLIP = 15 * video_data['video_info']['fps']
            
            # Get either AI hits or CLIP indices
            source_indices = sorted(list(set().union(*all_detections)))
            if not source_indices:
                raw_indices = video_data.get("original_indices", [])
                source_indices = sorted([int(x) for x in raw_indices])

            if source_indices:
                # group_with_gaps returns lists of frames; we convert to (min, max)
                groups = group_with_gaps(source_indices, max_gaps=MAX_GAPS_CLIP)
                foi = [(min(g), max(g)) for g in groups]
            else:
                foi = [-1]

        if foi == [-1]:
            if measure_metrics:
                log_metrics("num_model_checks", _model_check_count)
                log_metrics("num_vlm_detections", _vlm_detection_count)
                if config.adaptive_gt:
                    time_metrics["gt_frames"] = sorted(list(gt_indices))
                return foi, all_detections, [-1], time_metrics
            return foi, all_detections, [-1], None

        final_ranges = []
        fps = video_data['video_info']['fps']
        frame_count = video_data['video_info']["frame_count"]
        MAX_GAPS = BRIDGE_MULT * fps
        tw_before, tw_after = resolve_target_window(target_window)

        for foi_group in foi:
            group_min, group_max = min(foi_group), max(foi_group)
            
            # Calculate boundaries with padding
            start_ext = max(0, int(group_min + tw_before * fps))
            end_ext = min(frame_count - 1, int(group_max + tw_after * fps))
            
            # Instead of update(range(...)), we just store the boundary
            final_ranges.append((start_ext, end_ext)) 
        final_ranges.sort()

        merged = []
        if final_ranges:
            curr_start, curr_end = final_ranges[0]
            for next_start, next_end in final_ranges[1:]:
                # If the next range starts before the current one ends (plus MAX_GAPS)
                if next_start <= curr_end + MAX_GAPS:
                    curr_end = max(curr_end, next_end)
                else:
                    merged.append((curr_start, curr_end))
                    curr_start, curr_end = next_start, next_end
            merged.append((curr_start, curr_end))

        if True:
            logging.info("\n" + "-"*107)
            logging.info(f"Automaton_foi {automaton_foi}")
            logging.info(f"All Detections: {all_detections}")
            logging.info(f"Detected frames of interest: {foi}")
            logging.info(f"Merged Frames of Interests: {merged}")

        if measure_metrics:
            log_metrics("num_model_checks", _model_check_count)
            log_metrics("num_vlm_detections", _vlm_detection_count)
            if config.adaptive_gt:
                    time_metrics["gt_frames"] = sorted(list(gt_indices))
                    print(f'[DEBUG] Ground Truth indices: {time_metrics["gt_frames"]}')
            return foi, all_detections, merged, time_metrics
        return foi, all_detections, merged, None
    
    else:
        if PRINT_ALL:
            print(f"Automaton indices: {automaton_foi}")

        # if not automaton_foi or not any(len(x) > 0 for x in all_detections):
        if not automaton_foi: # automaton empty or nothing detected
            foi = [-1]
        else:
            fps = video_data['video_info']['fps']
            detections_foi = intersection_with_gaps(all_detections, max_gaps=fps * CONTEXT_SECONDS)
            print(detections_foi)
            detections_foi = list(range(int(min(detections_foi)), int(max(detections_foi)) + 1))
            if PRINT_ALL:
                print(f"Detection indices: {detections_foi}")

            foi = list(set(automaton_foi) & set(detections_foi)) # set intersection
            if len(foi) == 0:
                foi = [-1]
            else:
                foi = [min(foi), max(foi)]
        
        if foi == [-1]:
            print("[WARNING] Automaton Retuned Empty!")
            print("\n" + "-"*107)
            print(f"Automaton_foi {[]}")
            print(f"All Detections: {all_detections}")
            print(f"Detected frames of interest: {[-1]}")
            print(f"Merged Frames of Interests: {[-1]}")
            if measure_metrics:
                log_metrics("num_model_checks", _model_check_count)
                log_metrics("num_vlm_detections", _vlm_detection_count)
                if config.adaptive_gt:
                    time_metrics["gt_frames"] = sorted(list(gt_indices))
                    print(f'[DEBUG] Ground Truth indices: {time_metrics["gt_frames"]}')
                return foi, all_detections, [-1], time_metrics
            return [-1], all_detections, [-1], None

        fps = video_data['video_info']['fps']
        frame_count = video_data['video_info']["frame_count"]
        MAX_GAPS = BRIDGE_MULT * fps
        tw_before, tw_after = resolve_target_window(target_window)
        final_ranges = []
        
        if not isinstance(foi[0], list):
            foi = [foi]
        for foi_group in foi:
            group_min, group_max = min(foi_group), max(foi_group)
            
            # Calculate boundaries with padding
            start_ext = max(0, int(group_min + tw_before * fps))
            end_ext = min(frame_count - 1, int(group_max + tw_after * fps))
            
            # Instead of update(range(...)), we just store the boundary
            final_ranges.append((start_ext, end_ext))
            
        final_ranges.sort()
        merged = []
        if final_ranges:
            curr_start, curr_end = final_ranges[0]
            for next_start, next_end in final_ranges[1:]:
                # If the next range starts before the current one ends (plus MAX_GAPS)
                if next_start <= curr_end + MAX_GAPS:
                    curr_end = max(curr_end, next_end)
                else:
                    merged.append((curr_start, curr_end))
                    curr_start, curr_end = next_start, next_end
            merged.append((curr_start, curr_end))

        if True:
            print("\n" + "-"*107)
            print(f"Automaton_foi {automaton_foi}")
            print(f"All Detections: {all_detections}")
            print("Detected frames of interest:")
            print(foi)
            print(f"Merged Frames of Interests: {merged}")

        if measure_metrics:
            log_metrics("num_model_checks", _model_check_count)
            log_metrics("num_vlm_detections", _vlm_detection_count)
            if config.adaptive_gt:
                    time_metrics["gt_frames"] = sorted(list(gt_indices))
                    print(f'[DEBUG] Ground Truth indices: {time_metrics["gt_frames"]}')
            return foi, all_detections, merged, time_metrics
        
        vlm.clear_gpu_memory()
        return foi, all_detections, merged, None

