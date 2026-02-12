import json
import os
import subprocess
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse
from functools import partial
import shutil
from datetime import datetime
import re
import uuid

def time_to_seconds_srt(time_str):
    """Converts SRT timestamp 00:00:02,090 to seconds."""
    time_str = time_str.replace(" ", "").replace(",", ".")
    parts = time_str.split(":")
    h = int(parts[0])
    m = int(parts[1])
    s = float(parts[2])
    return h * 3600 + m * 60 + s

def get_video_duration(video_path):
    try:
        duration_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                        "-of", "csv=p=0", video_path]
        result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return float("inf")

def burn_subtitles_on_video(video_path, subtitle_srt_path, save_path, out_dir):
    log_file = os.path.join(out_dir, "err.txt")
    video_id = os.path.basename(video_path)
    
    # 1. Check if subtitle file exists; if not, just copy the original video
    if not os.path.exists(subtitle_srt_path) or os.path.getsize(subtitle_srt_path) == 0:
        with open(log_file, "a") as f:
            f.write(f"[INFO] [{video_id}] Subtitle missing or empty. Copying original.\n")
        shutil.copy(video_path, save_path)
        return

    # 2. Escape the path for the FFmpeg 'subtitles' filter
    # FFmpeg requires ':' to be escaped as '\:' and '\' as '/'
    safe_subtitle_path = subtitle_srt_path.replace("\\", "/").replace(":", "\\:")
    
    style = "PrimaryColour=&HFFFFFF&,BackColour=&H000000&,BorderStyle=3,Outline=1,Shadow=1,FontSize=16"
    
    # 3. Construct the command
    # Using 'ultrafast' preset to minimize compute time on the HPC nodes
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"subtitles='{safe_subtitle_path}':force_style='{style}'",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "copy", "-y", save_path
    ]

    try:
        # 4. Execute with a generous timeout (1 hour) to avoid the 10-minute cutoff
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)
        
    except subprocess.TimeoutExpired:
        with open(log_file, "a") as f:
            f.write(f"[TIMEOUT] [{video_id}] Process killed after 1 hour limit.\n")
        shutil.copy(video_path, save_path)
        
    except subprocess.CalledProcessError as e:
        with open(log_file, "a") as f:
            # Capturing the end of the log to see the actual error message
            f.write(f"[BURN ERROR] [{video_id}] Exit {e.returncode}\n{e.stderr}\n")
        
        # Ensure save_path exists even if FFmpeg failed
        if not os.path.exists(save_path):
            shutil.copy(video_path, save_path)

def process_entry(item, data_dir, out_dir):
    """
    Adapted for Video-MME item structure.
    Expects item to have 'video_id' and 'duration' (short/medium/long).
    """
    video_id = item["videoID"]
    duration_group = item.get("duration", "") # short/medium/long
    
    # Path logic: Video-MME usually puts videos in duration folders
    # Subtitles are usually in one flat folder or duration-based
    video_path = os.path.join(data_dir, "data", f"{video_id}.mp4")
    subtitle_path = os.path.join(data_dir, "subtitle", f"{video_id}.srt")
    
    # Ensure output subdirectory exists for the duration group
    os.makedirs(os.path.join(out_dir, duration_group), exist_ok=True)
    save_path = os.path.join(out_dir, duration_group, f"{video_id}.mp4")

    if os.path.exists(save_path):
        return

    if not os.path.exists(video_path):
        # Fallback if duration folders aren't used in your local extraction
        video_path = os.path.join(data_dir, "videos", f"{video_id}.mp4")

    if not os.path.exists(video_path):
        with open(os.path.join(out_dir, "err.txt"), "a") as error_log:
            error_log.write(f"Video file not found: {video_id}\n")
        return

    burn_subtitles_on_video(video_path, subtitle_path, save_path, out_dir)

def main(data_dir, out_dir, data, categories=["Temporal Reasoning"]):
    # Video-MME is large; keeping parallel workers restricted to prevent IO bottleneck
    num_workers = min(10, cpu_count())
    print(f"Using {num_workers} parallel workers for Video-MME.")
    
    # Video-MME has multiple questions per video_id. 
    # We should unique the data by video_id to avoid redundant processing.
    unique_videos = {item['videoID']: item for item in data}

    temporal_videos = []
    for key, video in unique_videos.items():
        if video["task_type"] in categories:
            video["video_path"] = f"{out_dir}/{video["duration"]}/{video["videoID"]}.mp4"
            temporal_videos.append(video)
    
    with open(os.path.join(args.out_dir, "dataset.json"), "w") as f:
        json.dump(temporal_videos, f, indent=4)

    partial_func = partial(process_entry, data_dir=data_dir, out_dir=out_dir)
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(partial_func, temporal_videos), total=len(temporal_videos)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path containing 'videos' and 'subtitles'")
    parser.add_argument("--out_dir", type=str, required=True, help="Where to save burned videos")

    parser.add_argument('--categories', nargs='+', type=str, default=["Temporal Reasoning"])
    args = parser.parse_args()
    print(f"[INFO] Burning subtitles to the following categories {args.categories}")
    # Load Video-MME metadata (usually test.json)
    # The JSON structure is: [{"video_id": "001", "duration": "short", ...}, ...]
    with open(os.path.join(args.data_dir, "dataset.json"), "r") as f:
        data = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    main(data_dir=args.data_dir, out_dir=args.out_dir, categories=args.categories, data=data)