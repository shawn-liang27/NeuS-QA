# import torch
# from decord import VideoReader, cpu
# from sentence_transformers import SentenceTransformer, util
# from PIL import Image
# import numpy as np
# import tqdm
# import logging

# CLIP_SAMPLE_RATE = 0.5

# def get_relevant_frames_from_video(model, video_path, propositions, threshold=0.2, sample_rate=1, window_padding=1.0):
#     """
#     Trims a long video into relevant chunks based on text propositions.
    
#     Args:
#         video_path: Path to the mp4 file.
#         propositions: List of strings (e.g., ["man_in_suit", "subtitle_hello"]).
#         threshold: Cosine similarity score above which a frame is 'relevant'.
#         window_padding: Seconds to add before and after a hit for context.
    
#     Returns:
#         {
#             "images": list[numpy.array],
#             "original_indices": list[int],
#             "video_info": 
#                 {
#                     "fps": fps,
#                     "frame_count": frame_count,
#                     "sample_rate" : int,
#                 }
#         }

#     """

#     vr = VideoReader(video_path, ctx=cpu(0))
#     fps = vr.get_avg_fps()
#     frame_count = len(vr)
#     cleaned_props = []
#     for p in propositions:
#         clean = p.replace("subtitle_", "").replace("_", " ")
#         cleaned_props.append(clean)

#     prop_embeddings = model.encode(cleaned_props, convert_to_tensor=True)
        
#     scan_step = int(max(1, fps / CLIP_SAMPLE_RATE)) 
#     coarse_indices = list(range(0, frame_count, scan_step))
    
#     relevant_segments = []
    
#     coarse_frames = vr.get_batch(coarse_indices).asnumpy()

#     print(f"[DEBUG] Video {video_path} FPS {fps} Length {frame_count} Scanning {len(coarse_indices)} frames for {len(cleaned_props)} Proposition...\nSampled Frame: {coarse_indices}")
#     with torch.no_grad():
#         looper = tqdm.tqdm(enumerate(coarse_indices), total=len(coarse_indices))
#         for i, idx in looper:
#             # frame = Image.fromarray(vr[idx].asnumpy())
#             frame = Image.fromarray(coarse_frames[i])
#             frame_emb = model.encode(frame, convert_to_tensor=True, show_progress_bar=False)
            
#             # Compute cosine similarity: [num_cleaned_props]
#             similarities = util.cos_sim(frame_emb, prop_embeddings)[0]
            
#             # If any proposition matches above threshold
#             if torch.any(similarities > threshold):
#                 timestamp = idx / fps
#                 # Define a window around this timestamp
#                 start = max(0, timestamp - window_padding)
#                 end = min(frame_count/fps, timestamp + window_padding)
#                 relevant_segments.append([start, end])

#     # 5. Merge Overlapping Windows
#     # (prevents InternVL2 from seeing the same 5-second clip 10 times)
#     print(f'[DEBUG] Number of Frames Kept by CLIP: {len(relevant_segments)}')

#     if not relevant_segments:
#         scan_step = int(max(1, fps / sample_rate)) 
#         normal_sample_indices = list(range(0, frame_count, scan_step))        
#         normal_sample_frames = vr.get_batch(normal_sample_indices).asnumpy()
#         logging.info("[WARNING] CLIP Relevant Search Speedup Failed, Switching Back to Uniform Sampling")
#         return {
#             "images": normal_sample_frames,
#             "original_indices": normal_sample_indices,
#             "sample_rate" : sample_rate,
#             "video_info": {"fps": fps, "frame_count": frame_count}
#         }

#     relevant_segments.sort(key=lambda x: x[0])
#     merged = [relevant_segments[0]]
#     for current in relevant_segments[1:]:
#         prev = merged[-1]
#         if current[0] <= prev[1]: 
#             prev[1] = max(prev[1], current[1])
#         else:
#             merged.append(current)

#     print(f'[DEBUG] Final CLIP Video Segments Chosen: {merged}')

#     all_indices_to_load = []
#     for start_t, end_t in merged:
#         duration = end_t - start_t
#         start_idx = int(start_t * fps)
#         end_idx = int(min(frame_count - 1, end_t * fps))
        
#         num_samples = max(3, int(duration * sample_rate))        # Ensure at least one full window
        
#         if duration < 1.0:
#             num_samples = 3

#         chunk_idxs = np.linspace(start_idx, end_idx, num=num_samples, dtype=int)
#         all_indices_to_load.extend(chunk_idxs.tolist())

#     # Sort and remove duplicates to maintain perfect chronological order
#     final_indices = sorted(list(set(all_indices_to_load)))

#     # Decord get_batch is most efficient when called once
#     flat_frames = vr.get_batch(final_indices).asnumpy()
#     # Return as a simple list of RGB arrays
#     return {
#         "images": [f for f in flat_frames],
#         "original_indices": final_indices,
#         "sample_rate" : sample_rate,
#         "video_info": {"fps": fps, "frame_count": frame_count}
#     }



import torch
import numpy as np
import tqdm
from decord import VideoReader, cpu
from sentence_transformers import util
from PIL import Image

def get_relevant_frames_from_video(model, video_path, propositions, 
                               threshold=0.22, sample_rate= 1, top_k=200, 
                               merge_threshold_sec=3.0,
                               window_padding = 1.0, 
                               max_frames_per_segment=50):
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

    # --- 1. COARSE SCAN (0.5 FPS) ---
    CLIP_SCAN_HZ = 0.5
    UNIFORM_TOTAL_FRAMES=210
    scan_step = int(max(1, fps / CLIP_SCAN_HZ))
    coarse_indices = list(range(0, frame_count, scan_step))
    coarse_frames = vr.get_batch(coarse_indices).asnumpy()
    
    cleaned_props = []
    for p in propositions:
        clean = p.replace("subtitle_", "").replace("_", " ")
        cleaned_props.append(clean)

    print(f"[DEBUG] Video {video_path} FPS {fps} Length {frame_count} Scanning {len(coarse_indices)} frames for {len(cleaned_props)} Proposition...\nSampled Frame: {coarse_indices}")

    prop_embeddings = model.encode(cleaned_props, convert_to_tensor=True)
    frame_scores = []

    with torch.no_grad():
        looper = tqdm.tqdm(enumerate(coarse_indices), total=len(coarse_indices))
        for i, idx in looper:
            # Encode frame and get best similarity across all propositions
            frame_emb = model.encode(Image.fromarray(coarse_frames[i]), convert_to_tensor=True, show_progress_bar=False)
            similarities = util.cos_sim(frame_emb, prop_embeddings)[0]
            max_sim = float(torch.max(similarities))
            
            if max_sim > threshold:
                frame_scores.append({"idx": idx, "score": max_sim})

    # --- 2. TOP-K ANCHOR SELECTION ---
    frame_scores.sort(key=lambda x: x["score"], reverse=True)
    top_anchors = frame_scores[:top_k]

    if not top_anchors:
        print(f"[DEBUG] Zero hits at {threshold}. Falling back to Top-50 anchors above 0.18.")
        top_anchors = [s for s in frame_scores if s["score"] >= 0.18][:top_k]

    if not top_anchors:
        print("[WARNING] Total CLIP failure. Using 0.1 FPS Global Fallback.")
        
    if not top_anchors:
        logging.info("[WARNING] CLIP Relevant Search Speedup Failed, Switching Back to Uniform Sampling")
        indices = np.linspace(0, frame_count - 1, num=UNIFORM_TOTAL_FRAMES, dtype=int)    
        frames = vr.get_batch(indices).asnumpy()
        return {
            "images": frames,
            "original_indices": indices,
            "sample_rate" : sample_rate,
            "video_info": {"fps": fps, "frame_count": frame_count}
        }

    # Convert anchors to time-windows (1s padding each side)
    raw_segments = []
    for anchor in top_anchors:
        t = anchor["idx"] / fps
        raw_segments.append([max(0, t - window_padding), min(duration_total, t + window_padding)])

    print(f'[DEBUG] Number of Frames Kept by CLIP: {raw_segments}\nLength: {len(raw_segments)}')
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

    print(f'[DEBUG] Final CLIP Video Segments Chosen: {merged}')

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
    final_indices = sorted(list(set(final_indices)))
    flat_frames = vr.get_batch(final_indices).asnumpy()
    print(f'CLIP Complete!\nRetained Indices: {final_indices}\nLength: {len(final_indices)}')
    return {
        "images": [f for f in flat_frames],
        "original_indices": final_indices,
        "video_info": {"fps": fps, "frame_count": frame_count}
    }