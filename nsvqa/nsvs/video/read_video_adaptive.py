import torch
import numpy as np
import tqdm
from decord import VideoReader, cpu
from sentence_transformers import util
from PIL import Image
import logging


def read_video_adaptive(model, video_path, propositions, 
                               threshold=0.23, tau_r=0.95, sample_rate=1, top_k=200, 
                               merge_threshold_sec=3.0, window_padding=1.0, 
                               max_frames_per_segment=50, uniform_sample_size_limit=240):
    """
    Refined Pipeline:
    1. Coarse CLIP Scan.
    2. Top-K Anchor Selection (Budgeting).
    3. Temporal Proximity Merging (Narrative bridging).
    4. Adaptive Square-Root Sampling (Density control).
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    fps, frame_count = vr.get_avg_fps(), len(vr)
    duration_total = frame_count / fps

    expected_uniform_count = int(duration_total * sample_rate)
    
    if expected_uniform_count <= uniform_sample_size_limit:
        logging.info(f"[DEBUG] Video {video_path} is within budget ({expected_uniform_count} frames). Using Uniform Sampling.")
        # Uniform sampling logic
        step = int(max(1, fps / sample_rate))
        indices = list(range(0, frame_count - 1, step))
        frames = vr.get_batch(indices).asnumpy()
        return {
            "images": [f for f in frames],
            "original_indices": indices,
            "video_info": {"fps": fps, "frame_count": frame_count, "sample_rate": sample_rate}
        }

    # --- TIER 2: CLIP RELEVANCE PIPELINE (For Long Videos) ---
    logging.info(f"[DEBUG] Video {video_path} exceeds budget. Initiating CLIP filtering.")

    # --- 1. COARSE SCAN (0.5 FPS) ---
    CLIP_SCAN_HZ = 0.5
    scan_step = int(max(1, fps / CLIP_SCAN_HZ))
    coarse_indices = list(range(0, frame_count - 1, scan_step))
    coarse_frames = vr.get_batch(coarse_indices).asnumpy()
    
    cleaned_props = []
    for p in propositions:
        clean = p.replace("subtitle_", "").replace("_", " ")
        cleaned_props.append(clean)

    print(f"[DEBUG] Video {video_path} FPS {fps} Length {frame_count} Scanning {len(coarse_indices)} frames for {len(cleaned_props)} Proposition...\nSampled Frame: {coarse_indices}")
    
    print(f'[DEBUG] Max coarse index: {max(coarse_indices)}\nFrame Count: {frame_count}')
    prop_embeddings = model.encode(cleaned_props, convert_to_tensor=True)
    frame_scores = []
    
    with torch.no_grad():
        coarse_embs = model.encode([Image.fromarray(f) for f in coarse_frames], 
                                   convert_to_tensor=True, show_progress_bar=True)
        
        for i, idx in enumerate(coarse_indices):
            similarities = util.cos_sim(coarse_embs[i], prop_embeddings)[0]
            max_sim = float(torch.max(similarities))
            if max_sim > threshold:
                frame_scores.append({"idx": idx, "score": max_sim})

    # --- 2. TOP-K ANCHOR SELECTION ---
    frame_scores.sort(key=lambda x: x["score"], reverse=True)
    top_anchors = frame_scores[:top_k]

    if not top_anchors:
        logging.info(f"[WARNING] Zero hits at {threshold}. Falling back to Top-50 anchors above 0.18.")
        top_anchors = [s for s in frame_scores if s["score"] >= threshold-0.03][:top_k]
        
    if not top_anchors:
        logging.info("[WARNING] CLIP Relevant Search Speedup Failed, Switching Back to Uniform Sampling")
        uniform_sample_count = int(duration_total * sample_rate)

        if uniform_sample_count > uniform_sample_size_limit:
            logging.info(f"[WARNING] Video {video_path} not is within budget ({uniform_sample_count} frames). Using Uniform Samplin Limit: {uniform_sample_size_limit}.")
            uniform_sample_count = uniform_sample_size_limit

        indices = np.linspace(0, frame_count - 1, num=uniform_sample_size_limit, dtype=int).tolist()   
        frames = vr.get_batch(indices).asnumpy()
        return {
            "images": frames,
            "original_indices": indices,
            "video_info": {"fps": fps, "frame_count": frame_count, "sample_rate" : sample_rate}
        }

    # Convert anchors to time-windows (1s padding each side)
    raw_segments = []
    for anchor in top_anchors:
        t = anchor["idx"] / fps
        raw_segments.append([max(0, t - window_padding), min(duration_total, t + window_padding)])

    logging.info(f'[DEBUG] Number of Frames Kept by CLIP: {raw_segments}\nLength: {len(raw_segments)}')
    # --- 3. GRACEFUL MERGING ---
    # Merge segments that are within x seconds of each other to create continuous 'scenes'
    raw_segments.sort(key=lambda x: x[0])
    merged = []
    if raw_segments:
        curr_s, curr_e = raw_segments[0]
        for next_s, next_e in raw_segments[1:]:
            if next_s <= curr_e + merge_threshold_sec:
                curr_e = max(curr_e, next_e)
            else:
                merged.append([curr_s, curr_e])
                curr_s, curr_e = next_s, next_e
        merged.append([curr_s, curr_e])

    logging.info(f'[DEBUG] Final CLIP Video Segments Chosen: {merged}')

    # --- 4. ADAPTIVE SAMPLING ---
    # We use a non-linear (sqrt) growth to prevent long segments from exploding in frame count.
    final_indices = []
    for start_t, end_t in merged:
        segment_duration = end_t - start_t
        
        start_idx = int(start_t * fps)
        end_idx = int(min(frame_count - 1, end_t * fps))
        
        # Calculate how many frames we WOULD have at the requested sample_rate
        # If sample_rate is 1, a 10s segment wants 10 frames.
        requested_samples = int(segment_duration * sample_rate)
        
        # If the requested amount is within our 'safe' VLM budget, use it.
        # If it's too long, we 'compress' the segment into our max_frames_per_segment.
        if requested_samples <= max_frames_per_segment:
            num_samples = max(3, requested_samples)
        else:
            # This only triggers for very long segments, keeping them manageable
            num_samples = max_frames_per_segment
        
        # Uniformly pick the budget of frames from this specific scene
        chunk_idxs = np.linspace(start_idx, end_idx, num=num_samples, dtype=int)
        final_indices.extend(chunk_idxs.tolist())

    # Final chronological sort and retrieval
    raw_final_indices = sorted(list(set(final_indices)))
    final_frames_data = vr.get_batch(final_indices).asnumpy()

    logging.info(f'[DEBUG] CLIP Complete!\nRetained Indices: {raw_final_indices}\nLength: {len(raw_final_indices)}')

    with torch.no_grad():
        final_embs = model.encode([Image.fromarray(f) for f in final_frames_data], 
                                  convert_to_tensor=True, show_progress_bar=True)

    deduplicated_indices = [raw_final_indices[0]]
    last_kept_idx = 0

    for i in range(1, len(raw_final_indices)):
        similarity = util.cos_sim(final_embs[last_kept_idx], final_embs[i])
        
        # Keep the frame only if it's different enough from the last kept one
        if similarity < tau_r:
            deduplicated_indices.append(raw_final_indices[i])
            last_kept_idx = i

    logging.info(f"Deduplication complete. Final count: {len(deduplicated_indices)} (from {len(raw_final_indices)})")

    final_frames = vr.get_batch(deduplicated_indices).asnumpy()
    return {
        "images": [f for f in final_frames],
        "original_indices": deduplicated_indices,
        "video_info": {"fps": fps, "frame_count": frame_count, "sample_rate": sample_rate}
    }