import json
import os
import subprocess
from typing import List, Dict, Any

from utils.video_cache import get_videos_to_process, register_videos, video_id_for_path, get_video_entry


SESSION_DIR = os.path.join("data", "session")
SELECTED_MANIFEST = os.path.join(SESSION_DIR, "selected_videos.json")
PROCESS_MANIFEST = os.path.join(SESSION_DIR, "videos_to_process.json")


def _ensure_directories() -> None:
    os.makedirs(os.path.join("data", "script"), exist_ok=True)
    os.makedirs(os.path.join("data", "audio"), exist_ok=True)
    os.makedirs(os.path.join("data", "matches"), exist_ok=True)
    os.makedirs(os.path.join("data", "metadata"), exist_ok=True)
    os.makedirs(os.path.join("outputs"), exist_ok=True)
    os.makedirs(SESSION_DIR, exist_ok=True)


def _load_selected_videos() -> List[Dict[str, Any]]:
    if os.path.exists(SELECTED_MANIFEST):
        with open(SELECTED_MANIFEST, "r", encoding="utf-8") as manifest_file:
            payload = json.load(manifest_file)
        videos = payload.get("videos", [])
        for video in videos:
            if video.get("path"):
                video["path"] = os.path.abspath(video["path"])
        return videos

    clips_folder = os.path.join("data", "clips")
    if not os.path.exists(clips_folder):
        return []

    videos: List[Dict[str, Any]] = []
    for filename in os.listdir(clips_folder):
        if filename.lower().endswith((".mp4", ".mov", ".mkv")):
            path = os.path.abspath(os.path.join(clips_folder, filename))
            videos.append(
                {
                    "id": video_id_for_path(path),
                    "path": path,
                    "filename": filename,
                }
            )

    if videos:
        os.makedirs(SESSION_DIR, exist_ok=True)
        with open(SELECTED_MANIFEST, "w", encoding="utf-8") as manifest_file:
            json.dump({"videos": videos}, manifest_file, indent=2)

    return videos


def main():
    _ensure_directories()

    script_path = os.path.join("data", "script", "narration_script.txt")
    audio_path = os.path.join("data", "audio", "narration_audio.mp3")
    metadata_folder = os.path.join("data", "metadata")
    matches_path = os.path.join("data", "matches", "matched_clips.json")
    output_path = os.path.join("outputs", "final_video.mp4")

    selected_videos = _load_selected_videos()
    video_paths = [video.get("path") for video in selected_videos if video.get("path")]

    if not video_paths:
        print("No videos found to process. Please select videos first.")
        return

    print("Videos to be analyzed:", video_paths)

    register_videos(video_paths)
    videos_to_process, _ = get_videos_to_process(video_paths)

    try:
        print("Generating AI narration...")
        subprocess.run(["python3", "src/scriptToTTS.py"], check=True)

        if videos_to_process:
            with open(PROCESS_MANIFEST, "w", encoding="utf-8") as manifest_file:
                json.dump({"videos": videos_to_process}, manifest_file, indent=2)

            print("Detecting objects in videos...")
            subprocess.run([
                "python3",
                "src/sceneDetection.py",
                "--manifest",
                PROCESS_MANIFEST,
            ], check=True)
        elif os.path.exists(PROCESS_MANIFEST):
            os.remove(PROCESS_MANIFEST)

        refreshed_videos: List[Dict[str, Any]] = []
        for video in selected_videos:
            video_id = video.get("id")
            cache_entry = get_video_entry(video_id) or {}
            updated = dict(video)
            if cache_entry.get("metadata_file"):
                updated["metadata_file"] = cache_entry["metadata_file"]
            if cache_entry.get("mtime") is not None:
                updated["mtime"] = cache_entry["mtime"]
            refreshed_videos.append(updated)

        selected_videos = refreshed_videos
        with open(SELECTED_MANIFEST, "w", encoding="utf-8") as manifest_file:
            json.dump({"videos": selected_videos}, manifest_file, indent=2)

        print("Matching script to clips...")
        subprocess.run(
            [
                "python3",
                "src/scriptMatching.py",
                script_path,
                metadata_folder,
                audio_path,
                matches_path,
                "--selected-videos",
                SELECTED_MANIFEST,
            ],
            check=True,
        )

        print("Starting video editing...")
        subprocess.run(
            [
                "python3",
                "src/videoProcessing.py",
                matches_path,
                audio_path,
                "data/clips",
                output_path,
                "--selected-videos",
                SELECTED_MANIFEST,
            ],
            check=True,
        )

        print("Video processing completed successfully!")

    except subprocess.CalledProcessError as error:
        print(f"Error during video processing: {error}")


if __name__ == "__main__":
    main()