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
import torch

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

def run_nsvs(
    video_data: dict,
    video_path: str,
    proposition: list,
    specification: str,
    target_window: str,
    model,
    device: int,
    measure_metrics: bool,
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
    def process_batch_windows(
    batch_pixels_cpu: torch.Tensor, 
    batch_indices: list[list[int]], 
    precomputed_inputs: dict, 
    proposition: list[str],
    threshold: float
    ):
        """
        Processes a batch of multiple windows at once.
        Returns: list[VideoFrame]
        """
        # 1. Run the optimized multi-window detection
        # This returns a flat list of responses (Length = Num Windows * Num Questions)
        batch_results = model.batch_detect_multi_window(
            scene_descriptions=proposition,
            batch_pixels_cpu=batch_pixels_cpu,
            precomputed_inputs=precomputed_inputs,
            batch_size=len(batch_indices),
            threshold=threshold
        )
        # print(f"RETURNED DetectedObject: {len(batch_results)}")
        
        num_questions = len(proposition)
        batch_video_frames = []

        results_list = []
        for window_indices, window_detections in zip(batch_indices, batch_results):
            # window_detections is already the correct list of DetectedObjects
            frame = VideoFrame(
                frame_idx_list=window_indices,
                object_of_interest={obj.name: obj for obj in window_detections}
            )
            results_list.append(frame)

        return results_list

    # def process_frame(sequence_of_frames: list[np.ndarray], current_indices: int, proposition: list):
    #     object_of_interest = {}    
    #     # Time per VLM proposition detection

    #     detected_objects = model.batch_detect(
    #                             batch_pixels_cpu=batch_pixels_cpu,
    #                             precomputed_inputs=precomputed_inputs,
    #                             scene_descriptions=proposition,
    #                             threshol=vlm_detection_thresholdd
    #                         )

    #     for detected_object in detected_objects:
    #         object_of_interest[detected_object.name] = detected_object      

    #     frame = VideoFrame(
    #         frame_idx_list=current_indices,
    #         object_of_interest=object_of_interest,
    #     )
    #     return frame

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
    frame_of_interest = FramesofInterest()

    if measure_metrics: 
        setup_duration = time.perf_counter() - setup_start
        log_metrics("automaton_set_up_time", setup_duration)


    BATCH_SIZE = 3

    frames_with_indices = [
        (img, idx) for img, idx in zip(video_data["images"], video_data["original_indices"])
    ]
    gap_thres = 3 * video_data['video_info']['fps']
    event_clusters = group_into_continuous_events(frames_with_indices, gap_threshold=240)

    frame_windows = []
    for event in event_clusters:
        for i in range(0, len(event), num_of_frame_in_sequence):
            window = event[i : i + num_of_frame_in_sequence]
            if len(window) == num_of_frame_in_sequence:
                frame_windows.append(window)

    images_to_process = []
    for window in frame_windows:
        for frame_data in window:
            images_to_process.append(frame_data[0])

    print(f"[DEBUG] Starting parallel preprocessing selected frames with dynamic patching")
    preprocess_vid_time = time.perf_counter() if measure_metrics else 0
    all_pixels, all_patches = model.load_video_from_seq_of_frames(seq_of_frames=images_to_process)
    patches_per_frame = all_patches[0]

    if measure_metrics: 
        preprocess_vid_duration = time.perf_counter() - preprocess_vid_time
        print(f'[DEBUG] Time Spent with InternVL2 dynamic patching {preprocess_vid_duration}')
        log_metrics("vlm_dynamic_patching_time", preprocess_vid_duration)
    
    if measure_metrics:
        log_metrics("num_frame_windows", len(frame_windows))

    window_patch_config = [patches_per_frame] * num_of_frame_in_sequence 
    pre_inputs = model.prepare_batch_inputs(proposition, window_patch_config)

    pixel_pointer = 0
    num_questions = len(proposition)
    patches_per_window = patches_per_frame * num_of_frame_in_sequence

    all_detections = [set(), set()]
    pbar = tqdm.tqdm(range(0, len(frame_windows), BATCH_SIZE), desc="Processing Batches")

    detect_result_frames = []
    for b_idx in pbar:
        batch_windows = frame_windows[b_idx : b_idx + BATCH_SIZE]
        actual_batch_size = len(batch_windows)

        print(f"\n[DEBUG] Batch Index: {b_idx}")
        print(f"[DEBUG] Number of windows in this batch: {len(batch_windows)}")
        print(f"[DEBUG] Global all_pixels shape: {all_pixels.shape}")
        print(f"[DEBUG] patches_per_frame: {patches_per_frame}")

        tiles_to_grab = actual_batch_size * patches_per_window
        batch_pixels_cpu = all_pixels[pixel_pointer : pixel_pointer + tiles_to_grab]
        pixel_pointer += tiles_to_grab

        batch_indices = [[f[1] for f in window] for window in batch_windows]

        print(f"Batch indices: {batch_indices}")
        print(f"DEBUG: GPU Pixel Shape: {batch_pixels_cpu.shape} | Batch Size: {len(batch_windows)}")
        # --- GPU INFERENCE ---
        # We pass the batch of windows. Internally, the model will extract features 
        # for all images once and expand them for the questions.
        per_window_detection_start = time.perf_counter() if measure_metrics else 0
        try:
            frames = process_batch_windows(
                batch_pixels_cpu=batch_pixels_cpu,
                batch_indices=batch_indices, # List of lists
                precomputed_inputs=pre_inputs,
                proposition=proposition,
                threshold=vlm_detection_threshold
            )
        except Exception as e:
            print(f"[FATAL] Caught Exception: {e}\nExiting the Program...")
            traceback.print_exc()
            sys.exit(1) 
        if measure_metrics: 
            per_batch_window_detection_duration = time.perf_counter() - per_window_detection_start
            log_metrics("per_batch_window_detection_time", per_batch_window_detection_duration, True)
            log_metrics("per_window_detection_time", per_batch_window_detection_duration / len(batch_indices), True)
        detect_result_frames.extend(frames)

    for frame in detect_result_frames:
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
        MAX_GAPS_CLIP = 3 * 60 * video_data['video_info']['fps']
        
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

    if foi != [-1]:
        fps = video_data['video_info']['fps']
        frame_count = video_data['video_info']["frame_count"]
        MAX_GAPS = BRIDGE_MULT * fps
        tw_before, tw_after = resolve_target_window(target_window)
        final_ranges = []
        
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

        # Return list of tuples: [(100, 500), (1200, 1500)]

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
        return foi, all_detections, merged, time_metrics

    return foi, all_detections, merged, None
    