from abc import ABC, abstractmethod
import subprocess
import json
import cv2

class Manager(ABC):
    @abstractmethod
    def load_data(self) -> list:
        pass
    
    @abstractmethod
    def postprocess_data(self, nsvs_path):
        pass


    def crop_video(self, entry, save_path, ground_truth=False):
        def group_into_ranges(frames, max_gap=1800): # 1800 frames = 1 minute at 30fps
            if not frames:
                return []
            frames = sorted(set(frames))
            ranges = []
            start = prev = frames[0]
            for f in frames[1:]:
                if f <= prev + max_gap:
                    prev = f
                else:
                    ranges.append((start, prev + 1))
                    start = prev = f
            ranges.append((start, prev + 1))
            return ranges

        input_path = entry["paths"]["video_path"]

        # Step 1: Determine frames to keep
        if ground_truth:
            positions = entry["metadata"]["position"]
            segments = []
            i = 0
            while i < len(positions) - 1:
                segments.append((positions[i], positions[i + 1]))
                i += 2
            if i == len(positions) - 1:
                center = positions[i]
                segments.append((center - 100, center + 100))
            foi_data = sorted(set(f for s, e in segments for f in range(max(0, s), e + 1)))
        else:
            foi_data = entry.get("frames_of_interest", [])
            if not foi_data or foi_data == [-1]:
                print(f"[Info] No specific interests found for {input_path}, exporting full video.")
                cap = cv2.VideoCapture(input_path)
                foi_data = list(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
                cap.release()
            
            if isinstance(foi_data[0], (list, tuple)):
                # We add +1 to the end because your FFmpeg trim uses end-exclusive logic
                ranges = [(s, e + 1) for s, e in foi_data]
            else:
                # Fallback for any old flat-list data
                ranges = group_into_ranges(foi_data)

        if not ranges:
            print(f"[Warning] No valid ranges for {input_path}, skipping.")
            return

        # Build filter_complex and output map
        filters = []
        labels = []
        for i, (start, end) in enumerate(ranges):
            filters.append(
                f"[0:v]trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS[v{i}]"
            )
            labels.append(f"[v{i}]")
        filters.append(f"{''.join(labels)}concat=n={len(ranges)}:v=1[outv]")

        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-filter_complex", "; ".join(filters),
            "-map", "[outv]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            save_path
        ]

        subprocess.run(cmd, check=True)

    def speed_adjust(self, entry, save_path, hardset=False):
        def group_into_ranges(frames):
            if not frames:
                return []
            frames = sorted(set(frames))
            ranges = []
            start = prev = frames[0]
            for f in frames[1:]:
                if f == prev + 1:
                    prev = f
                else:
                    ranges.append((start, prev + 1))
                    start = prev = f
            ranges.append((start, prev + 1))
            return ranges

        input_path = entry["paths"]["video_path"]

        # Determine important frames
        if hardset:
            positions = entry["metadata"]["position"]
            segments = []
            i = 0
            while i < len(positions) - 1:
                segments.append((positions[i], positions[i + 1]))
                i += 2
            if i == len(positions) - 1:
                center = positions[i]
                segments.append((center - 100, center + 100))
            important_frames = sorted(
                set(f for s, e in segments for f in range(max(0, s), e + 1))
            )
        else:
            important_frames = entry.get("frames_of_interest", [])
            if entry.get("nsvs", {}).get("output") == [-1]:
                cap = cv2.VideoCapture(input_path)
                important_frames = list(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
                cap.release()
            elif important_frames:
                start, end = min(important_frames), max(important_frames)
                important_frames = list(range(start, end + 1))

        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        all_frames = set(range(total_frames))
        imp_set = set(important_frames)
        unimp_set = all_frames - imp_set

        # compute speed factors so that
        #   output_duration_imp = 2/3 * total_output
        #   output_duration_unimp = 1/3 * total_output
        N_imp = len(imp_set)
        N_unimp = len(unimp_set)
        if total_frames == 0:
            return
        # speed factor = input_duration / desired_output_duration
        s_imp = N_imp / ((2/3) * total_frames)
        s_unimp = N_unimp / ((1/3) * total_frames)

        slow_ranges = group_into_ranges(imp_set)
        fast_ranges = group_into_ranges(unimp_set)

        filters = []
        labels = []
        idx = 0

        for start, end in sorted(slow_ranges + fast_ranges, key=lambda x: x[0]):
            if any(start >= s and end <= e for s, e in slow_ranges):
                # “important” segment: use s_imp
                rate = f"PTS/{s_imp:.6f}"
            else:
                # “unimportant” segment: use s_unimp
                rate = f"PTS/{s_unimp:.6f}"
            filters.append(
                f"[0:v]trim=start_frame={start}:end_frame={end},setpts={rate}[v{idx}]"
            )
            labels.append(f"[v{idx}]")
            idx += 1

        filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=1[outv]")

        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-filter_complex", "; ".join(filters),
            "-map", "[outv]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            save_path
        ]

        subprocess.run(cmd)

