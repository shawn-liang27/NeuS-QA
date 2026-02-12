import json
import os
import subprocess
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse
from functools import partial
import shutil
from datetime import datetime


now = datetime.now()
now_string = now.strftime("%Y_%m_%d_%H_%M")

def parse_subtitle_time(entry):
    """
    Normalizes subtitle entry to return start (seconds), end (seconds), and text.
    Handles None types safely.
    """
    if "timestamp" in entry and isinstance(entry["timestamp"], list):
        # Handle start time
        try:
            start = float(entry["timestamp"][0])
        except (ValueError, TypeError, IndexError):
            start = 0.0

        # Handle end time (Fix for your error)
        try:
            if len(entry["timestamp"]) > 1 and entry["timestamp"][1] is not None:
                end = float(entry["timestamp"][1])
            else:
                # Fallback: if end is None/missing, assume 2 seconds duration
                end = start + 2.0 
        except (ValueError, TypeError):
            end = start + 2.0

        text = entry.get("text", "")
        return start, end, text

    # SCHEMA 1: String timestamps
    elif "start" in entry and "end" in entry:
        start = time_to_seconds(entry["start"])
        end = time_to_seconds(entry["end"])
        text = entry.get("line", "")
        return start, end, text
        
    return None, None, None

def time_to_seconds(time_str):
    time_str = time_str.replace(" ", "")
    t = datetime.strptime(time_str, "%H:%M:%S.%f")
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6

def time_to_seconds(time_str):
    # Handle "00:00:01,500" vs "00:00:01.500"
    time_str = time_str.replace(" ", "").replace(",", ".")
    try:
        parts = time_str.split(":")
        h = int(parts[0])
        m = int(parts[1])
        s = float(parts[2])
        return h * 3600 + m * 60 + s 
    except (ValueError, IndexError):
        return 0.0


def seconds_to_srt_format(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def burn_subtitles_on_video(video_path, subtitles_json_path, starting_timestamp, save_path, out_dir,
                             font_size=24, font="Arial-Bold", color="white"):
    with open(subtitles_json_path, "r") as f:
        subtitles = json.load(f)

    if not subtitles:
        with open(f"{out_dir}/err.txt", "a") as error_log:
            error_log.write(f"empty subtitle file for video: {video_path}\n, subtitle path read: {subtitles_json_path}")
        return
    
    try:
        duration_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                        "-of", "csv=p=0", video_path]
        result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
        video_duration = float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error getting video duration for {video_path}: {e}")
        video_duration = float("inf")

    temp_srt_path = f"{save_path}_temp.srt"

    parsed_entries = []
    for entry in subtitles:
        s, e, t = parse_subtitle_time(entry)
        if s is not None:
            parsed_entries.append({"start": s, "end": e, "text": t})
    
    bad_file = False
    for entry in parsed_entries:
        duration = entry["end"] - entry["start"]
        if abs(duration - 0.01) < 0.001 and entry["text"] != "[Music]":
            bad_file = True
    if bad_file:
        with open(f"{out_dir}/err.txt", "a") as error_log:
            error_log.write(f"Bad subtitles detected for video: {video_path}\n")

    modified_subtitles = []
    for entry in parsed_entries:

        start = entry["start"] + starting_timestamp
        if bad_file:
            end = start + 1.5
        else:
            end = entry["end"] + starting_timestamp
        if start >= video_duration:
            continue
            
        end = min(end, video_duration)
        modified_subtitles.append({"start": start, "end": end, "line": entry["text"]})

    if not modified_subtitles:
        with open(f"{out_dir}/err.txt", "a") as error_log:
            error_log.write(f"Skipping {video_path}: No valid subtitles found (List is empty). Video Duration {video_duration}" + "\n")
            try:
                shutil.copy(video_path, save_path)
                error_log.write(f"Copied original video to {save_path}")
            except Exception as copy_error:
                error_log.write(f"Failed to copy original video: {copy_error}\n")
        return

    with open(temp_srt_path, "w", encoding="utf-8") as out:
        for i, entry in enumerate(modified_subtitles, start=1):
            start_str = seconds_to_srt_format(entry["start"])
            end_str = seconds_to_srt_format(entry["end"])
            out.write(f"{i}\n{start_str} --> {end_str}\n{entry['line']}\n\n")

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"subtitles={temp_srt_path}:force_style='PrimaryColour=&HFFFFFF&,BackColour=&H000000&,BorderStyle=3'",
        "-c:a", "copy", "-y", save_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg error for {video_path}:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
        with open(f"{out_dir}/err.txt", "a") as error_log:
            error_log.write(error_msg + "\n")
            if e.stderr:
                try:
                    error_log.write(e.stderr.decode("utf-8", errors="ignore") + "\n")
                except AttributeError:
                    error_log.write(str(e.stderr) + "\n")
            
            # copy the original video to the save path
            try:
                shutil.copy(video_path, save_path)
                print(f"Copied original video to {save_path}")
            except Exception as copy_error:
                error_log.write(f"Failed to copy original video: {copy_error}\n")
    finally:
        try:
            os.remove(temp_srt_path)
        except FileNotFoundError:
            pass


def process_entry(entry, data_dir, out_dir):
    video_id = entry["video_id"]
    video_file_name = entry["video_path"]
    video_subtitle_name = entry["subtitle_path"]
    video_path = f"{data_dir}/videos/{video_file_name}"
    subtitles_json_path = f"{data_dir}/subtitles/{video_subtitle_name}"
    save_path = f"{out_dir}/{video_file_name}"

    # Skip if already processed
    if os.path.exists(save_path):
        with open(f"{out_dir}/err.txt", "a") as error_log:
            print(f'Video \"{video_id}\" already exists at {video_path}')
        return

    if not os.path.exists(video_path) or not os.path.exists(subtitles_json_path):
        with open(f"{out_dir}/err.txt", "a") as error_log:
            if not os.path.exists(video_path):
                error_log.write(f"Video file not found: {video_path}\n")
            if not os.path.exists(subtitles_json_path):
                error_log.write(f"Subtitles file not found: {subtitles_json_path}\n")

        return

    try:
        starting_timestamp = entry["starting_timestamp_for_subtitles"]
    except KeyError as e:
        with open(f"{out_dir}/err.txt", "a") as error_log:
            error_log.write(f"KeyError: {e} for video_id: {video_id}\n")
        return

    burn_subtitles_on_video(video_path, subtitles_json_path, starting_timestamp, save_path, out_dir)


def main(data_dir, out_dir, data):
    num_workers = min(10, cpu_count())
    print(f"Using {num_workers} parallel workers.")
    partial_func = partial(process_entry, data_dir=data_dir, out_dir=out_dir)
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(partial_func, data), total=len(data)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument('--all', type=bool, default=False)
    parser.add_argument('--mix', type=bool, default=False)
    parser.add_argument('--categories', nargs='+', type=str, help='List of categories')
    args = parser.parse_args()
    
    categories = args.categories
    data_dir = args.data_dir
    out_dir = args.out_dir

    with open(f"{data_dir}/lvb_val.json", "r") as f:
        data = json.load(f)
    
    if args.all:
        print(f"Total Videos for LongVideoBenchEval: {len(data)}, output directory: {out_dir}")
        os.makedirs(out_dir, exist_ok=True)
        main(data_dir=data_dir, out_dir=out_dir, data=data)
        with open(f'{out_dir}/lvb_val.json', 'w') as f:
            json.dump(data, f, indent=4)

    elif args.mix:
        out_dir = f'{out_dir}/{"_".join(categories)}_mix_{now_string}'
        os.makedirs(out_dir, exist_ok=True)
        print(f"Total Videos for LongVideoBench Val: {len(data)}, output directory: {out_dir}")
        filtered_data = []
        for video in data:
            if video["question_category"] in categories:
                filtered_data.append(video)
        main(data_dir=data_dir, out_dir=out_dir, data=filtered_data)
        with open(f'{out_dir}/lvb_val.json', 'w') as f:
            json.dump(filtered_data, f, indent=4)
    else:
        for category in categories:
            out_dir = f'{out_dir}/{category}'
            os.makedirs(out_dir, exist_ok=True)
            filtered_data = []
            for video in data:
                if video["question_category"] == category:
                    filtered_data.append(video)

            os.makedirs(out_dir, exist_ok=True)
            print(f"Total Videos for Cateogry {category}: {len(filtered_data)}")
            main(data_dir=data_dir, out_dir=out_dir, data=filtered_data)
            with open(f'{out_dir}/lvb_val.json', 'w') as f:
                json.dump(filtered_data, f, indent=4)
