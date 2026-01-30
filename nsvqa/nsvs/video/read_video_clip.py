import torch
from decord import VideoReader, cpu
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import numpy as np

def get_relevant_video_chunks(model, video_path, propositions, prop_threshold=0.22, subtitle_threshold=0.18,window_padding=2.0):
    """
    Trims a long video into relevant chunks based on text propositions.
    
    Args:
        video_path: Path to the mp4 file.
        propositions: List of strings (e.g., ["man in suit", "subtitle_hello"]).
        threshold: Cosine similarity score above which a frame is 'relevant'.
        window_padding: Seconds to add before and after a hit for context.
    """

    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    
    cleaned_props = []
    for p in propositions:
        clean = p.replace("subtitle_", "").replace("_", " ")
        cleaned_props.append(clean)

    prop_embeddings = model.encode(propositions, convert_to_tensor=True)
    
    scan_step = int(max(1, fps / 2)) 
    sample_indices = list(range(0, len(vr), scan_step))
    
    relevant_segments = []
    
    all_sampled_frames = vr.get_batch(sample_indices).asnumpy()

    print(f"Scanning {len(sample_indices)} frames for {len(propositions)} propositions...")
    with torch.no_grad():
        for idx in sample_indices:

            # frame = Image.fromarray(vr[idx].asnumpy())
            frame = Image.fromarray(all_sampled_frames[i])
            frame_emb = model.encode(frame, convert_to_tensor=True)
            
            # Compute cosine similarity: [num_propositions]
            similarities = util.cos_sim(frame_emb, prop_embeddings)[0]
            
            # If any proposition matches above threshold
            for i, score in enumerate(similarities):
                current_prop = propositions[i]
                # Set lower threshold for subtitles because CLIP is weaker at OCR
                target_thresh = subtitle_threshold if "subtitle" in current_prop else prop_threshold
                if score > target_thresh:
                    timestamp = idx / fps
                    # Define a window around this timestamp
                    start = max(0, timestamp - window_padding)
                    end = min(len(vr)/fps, timestamp + window_padding)
                    relevant_segments.append([start, end])

            # if torch.any(similarities > threshold):
            #     timestamp = idx / fps
            #     # Define a window around this timestamp
            #     start = max(0, timestamp - window_padding)
            #     end = min(len(vr)/fps, timestamp + window_padding)
            #     relevant_segments.append([start, end])

    # 5. Merge Overlapping Windows
    # (prevents InternVL2 from seeing the same 5-second clip 10 times)
    if not relevant_segments:
        return []

    relevant_segments.sort(key=lambda x: x[0])
    merged = [relevant_segments[0]]
    for current in relevant_segments[1:]:
        prev = merged[-1]
        if current[0] <= prev[1]: 
            prev[1] = max(prev[1], current[1])
        else:
            merged.append(current)

    all_indices_to_load = []
    for start_t, end_t in merged:
        start_idx = int(start_t * fps)
        end_idx = int(min(len(vr) - 1, end_t * fps))
        
        # Determine how many frames to pull from this specific chunk
        # If NeuS-QA needs windows of 3-5, we sample at a rate that provides enough density
        num_samples = int((end_t - start_t) * 2) # e.g., 2 frames per second of relevance
        num_samples = max(num_samples, 5)        # Ensure at least one full window
        
        chunk_idxs = np.linspace(start_idx, end_idx, num=num_samples, dtype=int)
        all_indices_to_load.extend(chunk_idxs.tolist())

    # Sort and remove duplicates to maintain perfect chronological order
    final_indices = sorted(list(set(all_indices_to_load)))

    # 7. Batch load from disk (The Speedup)
    if not final_indices:
        return []
        
    # Decord get_batch is most efficient when called once
    flat_frames = vr.get_batch(final_indices).asnumpy()
    
    # Return as a simple list of RGB arrays
    return {
        "images": return [f for f in flat_frames],
        "original_indices": final_indices,
        # "sample_rate" : sample_rate,
        "video_info": {"fps": fps, "frame_count": frame_count}
    }
