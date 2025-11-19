import os
import json
import argparse
from typing import List, Dict, Any

import cv2
from ultralytics import YOLO

from utils.video_cache import register_videos, update_video_entry, video_id_for_path


yolo = YOLO('yolov8s.pt')


def _safe_fps(capture) -> int:
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps is None:
        return 0
    try:
        return int(fps)
    except (TypeError, ValueError):
        return 0


def detect_objects_for_videos(video_entries: List[Dict[str, Any]], output_folder: str) -> None:
    if not video_entries:
        print("No videos provided for analysis.")
        return

    os.makedirs(output_folder, exist_ok=True)

    for entry in video_entries:
        video_path = entry.get("path")
        if not video_path:
            print("Skipping entry without a video path.")
            continue

        absolute_path = os.path.abspath(video_path)
        if not os.path.exists(absolute_path):
            print(f"Video file not found: {absolute_path}")
            continue

        video_id = entry.get("id") or video_id_for_path(absolute_path)
        filename = entry.get("filename") or os.path.basename(absolute_path)
        metadata_path = entry.get("metadata_file") or os.path.join(output_folder, f"{video_id}_objects.json")

        print(f"Starting analysis for {filename} ({absolute_path})...")

        cap = cv2.VideoCapture(absolute_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {filename}")
            continue

        fps = _safe_fps(cap)
        if fps <= 0:
            print(f"Error: FPS is invalid for video {filename}. Skipping...")
            cap.release()
            continue

        metadata = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"End of video reached for {filename}")
                break

            if frame_count % fps == 0:
                results = yolo(frame)
                detected_objects = [
                    results[0].names[int(cls)]
                    for cls in results[0].boxes.cls.cpu().numpy()
                ]
                metadata.append({"timestamp": frame_count // fps, "objects": detected_objects})

            frame_count += 1

        cap.release()
        print(f"Finished analysis for {filename}. Frames analyzed: {frame_count // fps}")

        with open(metadata_path, "w", encoding="utf-8") as json_file:
            json.dump(metadata, json_file, indent=4)

        update_video_entry(video_id, metadata_file=metadata_path, mtime=os.path.getmtime(absolute_path))


def load_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        data = json.load(manifest_file)

    videos = data.get("videos", [])
    for video in videos:
        if "path" in video and video["path"]:
            video["path"] = os.path.abspath(video["path"])
        if "id" not in video or not video["id"]:
            video["id"] = video_id_for_path(video.get("path", ""))
        if "filename" not in video or not video["filename"]:
            video["filename"] = os.path.basename(video.get("path", ""))
        if "metadata_file" not in video or not video["metadata_file"]:
            video["metadata_file"] = os.path.join("data", "metadata", f"{video['id']}_objects.json")

    return videos


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze videos and cache object metadata.")
    parser.add_argument("clips_folder", nargs="?", default="data/clips", help="Fallback folder containing video clips")
    parser.add_argument("output_folder", nargs="?", default="data/metadata", help="Folder to write metadata files")
    parser.add_argument("--manifest", help="Optional JSON manifest describing videos to process")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    if args.manifest:
        video_entries = load_manifest(args.manifest)
        register_videos([entry["path"] for entry in video_entries if entry.get("path")])
    else:
        video_entries = []
        if os.path.exists(args.clips_folder):
            for filename in os.listdir(args.clips_folder):
                if filename.lower().endswith((".mp4", ".mov", ".mkv")):
                    video_path = os.path.join(args.clips_folder, filename)
                    video_entries.append(
                        {
                            "id": video_id_for_path(video_path),
                            "path": video_path,
                            "filename": filename,
                            "metadata_file": os.path.join(args.output_folder, f"{video_id_for_path(video_path)}_objects.json"),
                        }
                    )
        if not video_entries:
            print("No videos found to analyze.")

    detect_objects_for_videos(video_entries, args.output_folder)


if __name__ == "__main__":
    main()
