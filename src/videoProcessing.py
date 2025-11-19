import os
import json
import argparse
from typing import Any, Dict

from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip

def load_metadata(metadata_folder):
    """Load metadata for uploaded videos."""
    metadata = {}
    for file in os.listdir(metadata_folder):
        if file.endswith("_objects.json"):
            video_name = file.replace("_objects.json", ".mp4")
            with open(os.path.join(metadata_folder, file), "r") as f:
                metadata[video_name] = json.load(f)
    return metadata


def load_video_lookup(selected_videos_path: str) -> Dict[str, Dict[str, Any]]:
    if not selected_videos_path or not os.path.exists(selected_videos_path):
        return {}

    try:
        with open(selected_videos_path, "r", encoding="utf-8") as manifest_file:
            payload = json.load(manifest_file)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Unable to read selected videos manifest: {exc}")
        return {}

    lookup: Dict[str, Dict[str, Any]] = {}

    for video in payload.get("videos", []):
        video_id = video.get("id")
        if not video_id:
            continue
        lookup[video_id] = video

    return lookup


def edit_video_with_matches(matches_path, audio_path, clips_folder, output_path, selected_videos_path=None):
    """Create a video using the matched clips with synchronized audio."""

    # Step 1: Load matched clips
    if not os.path.exists(matches_path):
        print(f"Error: Matches file not found at {matches_path}")
        return

    with open(matches_path, "r") as file:
        matched_clips = json.load(file)

    if not matched_clips:
        print("No matched clips available.")
        return

    print(f"Loaded {len(matched_clips)} matched clips.")

    video_lookup = load_video_lookup(selected_videos_path)

    # Step 2: Load narration audio
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return

    narration_audio = AudioFileClip(audio_path)
    total_narration_duration = narration_audio.duration
    sentence_duration = total_narration_duration / len(matched_clips)

    print(f"Narration duration: {narration_audio.duration} seconds.")
    print(f"Each sentence will have a duration of {sentence_duration} seconds.")

    # Step 3: Create video clips from matches
    video_clips = []
    for index, match in enumerate(matched_clips):
        display_name = match.get("video") or match.get("video_id") or f"clip_{index}"
        video_file = match.get("video_path")
        if video_file:
            video_file = os.path.abspath(video_file)

        if (not video_file or not os.path.exists(video_file)) and match.get("video_id"):
            candidate = video_lookup.get(match["video_id"], {}).get("path")
            if candidate and os.path.exists(candidate):
                video_file = os.path.abspath(candidate)
                display_name = video_lookup.get(match["video_id"], {}).get("filename", display_name)

        if (not video_file or not os.path.exists(video_file)) and match.get("video"):
            candidate = os.path.join(clips_folder, match["video"])
            if os.path.exists(candidate):
                video_file = candidate

        if not video_file or not os.path.exists(video_file):
            print(f"Video file not found for match '{display_name}'. Skipping.")
            continue

        timestamp = match["timestamp"]

        print(f"Using video '{video_file}' for sentence: '{match['sentence']}'")
        try:
            clip = VideoFileClip(video_file).subclip(timestamp, timestamp + sentence_duration)
            video_clips.append(clip)
        except Exception as e:
            print(f"Error processing clip '{video_file}': {e}")
            continue

    if not video_clips:
        print("No video clips were created. Aborting.")
        return

    # Step 4: Concatenate video clips
    print("Concatenating video clips...")
    final_video = concatenate_videoclips(video_clips)

    # Step 5: Sync the narration audio with the video
    if narration_audio.duration > final_video.duration:
        print("Trimming audio to match video duration.")
        narration_audio = narration_audio.subclip(0, final_video.duration)

    final_video = final_video.set_audio(narration_audio)

    # Step 6: Export the final video
    print(f"Exporting video to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

    if os.path.exists(output_path):
        print(f"Video saved successfully to {output_path}")
    else:
        print("Error: Video file was not created.")

def parse_args():
    parser = argparse.ArgumentParser(description="Assemble a video using matched clips and narration audio.")
    parser.add_argument("matches_path", nargs="?", default="data/matches/matched_clips.json", help="Path to the matched clips JSON file")
    parser.add_argument("audio_path", nargs="?", default="data/audio/narration_audio.mp3", help="Path to the narration audio file")
    parser.add_argument("clips_folder", nargs="?", default="data/clips", help="Folder containing source video clips")
    parser.add_argument("output_path", nargs="?", default="outputs/final_video.mp4", help="Output path for the rendered video")
    parser.add_argument("--selected-videos", help="Path to selected videos manifest", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    edit_video_with_matches(
        args.matches_path,
        args.audio_path,
        args.clips_folder,
        args.output_path,
        selected_videos_path=args.selected_videos,
    )
