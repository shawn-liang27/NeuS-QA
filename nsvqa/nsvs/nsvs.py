from enum import auto
import numpy as np
import warnings
import tqdm
import os

from nsvqa.nsvs.model_checker.property_checker import PropertyChecker
from nsvqa.nsvs.model_checker.video_automaton import VideoAutomaton
from nsvqa.nsvs.video.frames_of_interest import FramesofInterest
from nsvqa.utils.intersection import intersection_with_gaps
from nsvqa.nsvs.video.video_frame import VideoFrame
from nsvqa.nsvs.vlm.vllm_client import VLLMClient

from nsvqa.nsvs.vlm.internvl import InternVL
import logging
import time
import sys

PRINT_ALL = False
warnings.filterwarnings("ignore")

def run_nsvs(
    video_data: dict,
    video_path: str,
    proposition: list,
    specification: str,
    model: str,
    device: int,
    vlm: str,
    measure_metrics: bool,
    model_type: str = "dtmc",
    num_of_frame_in_sequence = 3,
    tl_satisfaction_threshold: float = 0.6,
    detection_threshold: float = 0.5,
    vlm_detection_threshold: float = 0.349,
    image_output_dir: str = "output",
):
    """Find relevant frames from a video that satisfy a specification"""

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

    # vlm = VLLMClient(model=model, api_base=f"http://localhost:{device}/v1")

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

    frame_step = int(round(video_data["video_info"]["fps"] / video_data["sample_rate"]))
    frame_of_interest = FramesofInterest(num_of_frame_in_sequence, frame_step)

    frames = video_data["images"]

    frame_windows = []
    for i in range(0, len(frames), num_of_frame_in_sequence):
        frame_windows.append(frames[i : i + num_of_frame_in_sequence])

    def process_frame(sequence_of_frames: list[np.ndarray], frame_count: int, measure_metrics: bool):
        object_of_interest = {}
        for prop in proposition:     
            # Time Time per VLM proposition detection
            per_prop_detection_start = time.perf_counter() if measure_metrics else 0
            detected_object = vlm.detect(
                seq_of_frames=sequence_of_frames,
                scene_description=prop,
                threshold=vlm_detection_threshold
            )
            if measure_metrics: 
                per_prop_detection_duration = time.perf_counter() - per_prop_detection_start
                log_metrics("per_proposition_detection_time", per_prop_detection_duration, True) 
                nonlocal _vlm_detection_count
                _vlm_detection_count += 1

            object_of_interest[prop] = detected_object
            if PRINT_ALL and detected_object.is_detected:
                print(f"\t{prop}: {detected_object.confidence}->{detected_object.probability}")

        frame = VideoFrame(
            frame_idx=frame_count,
            frame_images=sequence_of_frames,
            object_of_interest=object_of_interest,
        )
        return frame

    if PRINT_ALL:
        looper = enumerate(frame_windows)
    else:
        looper = tqdm.tqdm(enumerate(frame_windows), total=len(frame_windows))

    if measure_metrics:
        log_metrics("num_frame_windows", len(frame_windows))
    all_detections = [[], []]
    for i, sequence_of_frames in looper:
        if PRINT_ALL:
            print("\n" + "*"*50 + f" {i}/{len(frame_windows)-1} " + "*"*50)
            print("Detections:")
        
        per_window_detection_start = time.perf_counter() if measure_metrics else 0

        try:
            frame = process_frame(sequence_of_frames, i, measure_metrics)
        except Exception as e:
            print(f"DEBUG: Caught {e}")

        if measure_metrics: 
            per_window_detection_duration = time.perf_counter() - per_window_detection_start
            log_metrics("per_frame_window_detection_time", per_window_detection_duration, True)

        if PRINT_ALL:
            print("Detections Completed and Returned")
        if PRINT_ALL and False: # disabled
            os.makedirs(image_output_dir, exist_ok=True)
            frame.save_frame_img(save_path=os.path.join(image_output_dir, f"{i}"))
        
        if checker.validate_frame(frame_of_interest=frame):
            # Time this number automaton addition
            thresh = frame.thresholded_detected_objects(threshold=detection_threshold)
            for prop in thresh.keys():
                split = checker.check_split(prop)
                if frame.frame_idx not in all_detections[split]:
                    all_detections[split].append(frame.frame_idx)
            if PRINT_ALL:
                print(f"\t{all_detections}")

            add_automaton_model_check_start = time.perf_counter() if measure_metrics else 0

            automaton.add_frame(frame=frame)
            frame_of_interest.frame_buffer.append(frame)
            # Time model checking time
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
        print(f"Automaton indices: {automaton_foi}")

    # if not automaton_foi or not any(len(x) > 0 for x in all_detections):
    if not automaton_foi: # automaton empty or nothing detected
        foi = [-1]
    else:
        detections_foi = [x * num_of_frame_in_sequence * frame_step for x in intersection_with_gaps(all_detections)]
        detections_foi = list(range(int(min(detections_foi)), int(max(detections_foi)) + 1))
        if PRINT_ALL:
            print(f"Detection indices: {detections_foi}")

        foi = list(set(automaton_foi) & set(detections_foi)) # set intersection
        if len(foi) == 0:
            foi = [-1]
        else:
            foi = [min(foi), max(foi)]

        if True:
            print("\n" + "-"*107)
            print(f"Automaton_foi: {automaton_foi}")
            print(f"All Detections: {all_detections}")
            print(f"Detected frames of interest: {detections_foi}")
            print(f"Merged Frames of Interests: {foi}")

    if measure_metrics:
        log_metrics("num_model_checks", _model_check_count)
        log_metrics("num_vlm_detections", _vlm_detection_count)
        return foi, all_detections, time_metrics
    vlm.clear_gpu_memory()
    return foi, all_detections, None

