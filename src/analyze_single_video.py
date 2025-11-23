#!/usr/bin/env python3
"""
Single video analyzer - called by the API server
Analyzes one video at a time with OpenAI Vision API
"""

import sys
import os
import json
import cv2
import base64
from io import BytesIO
from PIL import Image
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.utils.video_cache import update_video_entry, video_id_for_path
except ImportError:
    from utils.video_cache import update_video_entry, video_id_for_path

load_dotenv()

async def analyze_frame_async(client, img_base64, timestamp):
    """Analyze a single frame with OpenAI Vision API"""
    prompt = """You will be analyzing a single video frame in extreme detail.

### **Extract the Following Details Per Frame**
For the frame, extract the following details. If examples are provided in parenthesis, you MUST choose only from those choices.
1. **Description**: Give a detailed description of the frame (2-3 sentences).
2. **Main Subject**: Describe the main subject of the frame in detail.
3. **Setting**: Describe the setting/location/environment in detail.
4. **Time of Day**: Describe the time of day ("Morning", "Afternoon", "Evening", "Night", "Dawn", "Dusk")
5. **Camera Angle**: Describe the framing technique ("Wide", "Close-up", "Medium", "Extreme Close-up", "Regular", "Cropped", "Macro", "Micro", "Aerial", "Eye-level", "Low Angle", "High Angle").
6. **Orientation**: Describe the camera orientation and framing ("Horizontal", "Vertical", "Tilted", "Dutch Angle").
7. **Lighting**: Describe the lighting style in detail (e.g., "Natural", "Artificial", "Low Light", "Bright", "Backlit", "Side-lit", "Soft", "Hard", "Golden Hour", "Blue Hour", "Overcast", "Studio", "Dramatic", "Flat").
8. **Weather**: If applicable, describe weather conditions ("Sunny", "Cloudy", "Rainy", "Snowy", "Foggy", "Clear", "Stormy", "Overcast", "N/A").
9. **Motion**: Describe any motion or movement in the frame ("Static", "Slow", "Fast", "Panning", "Zooming", "Tracking", "Handheld", "Stable").
10. **Color Palette**: Describe the dominant colors and overall color mood (e.g., "Warm tones", "Cool blues", "Vibrant", "Muted", "Black and white", "Monochromatic").
11. **Mood/Atmosphere**: Describe the emotional tone or atmosphere ("Peaceful", "Tense", "Energetic", "Melancholic", "Joyful", "Mysterious", "Dramatic", "Calm").
12. **People**: List all people visible with their gender/age and what they're doing. Be SPECIFIC about gender (e.g., ["woman walking", "young girl playing", "man talking", "boy running", "elderly woman sitting", "teenage girl reading"]). Always include gender identifiers (woman/women, man/men, girl/girls, boy/boys, person if unclear).
13. **Animals**: List all animals visible (e.g., ["dog running", "bird flying", "cat sitting"]).
14. **Objects**: List ALL visible objects in detail - be exhaustive. Include furniture, vehicles, natural elements, buildings, tools, food, clothing items, electronics, etc. Return as a comprehensive list of strings.
15. **Actions/Activities**: List all activities or actions occurring (e.g., ["walking", "talking", "eating", "driving"]).
16. **Text Visible**: Extract ALL visible text, signs, labels, captions, subtitles, or words appearing anywhere in the frame. Be THOROUGH - capture every piece of text you can see, even if partially visible. Return as a list of strings. If there's absolutely no text, return an empty list. Examples: ["Welcome to Paris", "STOP", "Coca-Cola"], ["Recipe: 2 cups flour"], ["John Smith", "CEO"], ["Sale 50% Off"].
17. **Location Type**: Type of location ("Indoor", "Outdoor", "Studio", "Urban", "Rural", "Natural", "Industrial", "Residential", "Commercial").
18. **Scene Type**: Category of scene ("Landscape", "Portrait", "Action", "Dialogue", "Establishing Shot", "B-Roll", "Product Shot", "Close-up Detail").
19. **Geographic Context**: If you can identify or infer the location, country, region, or famous landmark from visible signs, architecture, landmarks, or context, list them (e.g., ["Eiffel Tower", "Paris", "France"] or ["Stockholm sign", "Sweden", "Scandinavia"] or ["El Capitan", "Yosemite", "California", "USA"]). Include both specific and broader geographic terms.
20. **Landmarks/Places**: Any recognizable landmarks, monuments, famous places, or named locations visible or inferable (e.g., ["Golden Gate Bridge", "Times Square", "Grand Canyon"]).

Your response must be a JSON object containing ALL fields above. Be extremely thorough and detailed, especially with the objects list - identify EVERYTHING visible.

Required JSON structure:
{
  "description": "string",
  "main_subject": "string",
  "setting": "string",
  "time_of_day": "string",
  "camera_angle": "string",
  "orientation": "string",
  "lighting": "string",
  "weather": "string",
  "motion": "string",
  "color_palette": "string",
  "mood": "string",
  "people": ["list of strings"],
  "animals": ["list of strings"],
  "objects": ["comprehensive list of all visible objects"],
  "actions": ["list of strings"],
  "text_visible": ["list of strings"],
  "location_type": "string",
  "scene_type": "string",
  "geographic_context": ["list of strings with locations, countries, regions"],
  "landmarks": ["list of strings with recognizable places"]
}"""

    max_retries = 5
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                            }
                        ]
                    }
                ],
                max_tokens=1500,
                response_format={"type": "json_object"}
            )

            frame_data = json.loads(response.choices[0].message.content)
            frame_data["timestamp"] = timestamp
            return frame_data, None

        except Exception as e:
            error_message = str(e)
            if "rate_limit" in error_message.lower() or "429" in error_message:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                    continue
            return None, str(e)

    return None, "Max retries exceeded"


async def process_video(video_path, video_id, metadata_path, thumbnails_folder, progress_callback=None):
    """Process a single video with OpenAI Vision API"""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise Exception("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")
    
    print(f"OpenAI API key found: {api_key[:10]}...", flush=True)
    client = AsyncOpenAI(api_key=api_key)

    if not os.path.exists(video_path):
        raise Exception(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        cap.release()
        raise Exception(f"Invalid FPS: {fps}")

    # Extract frames
    frames_to_process = []
    frame_count = 0
    
    print(f"Video FPS: {fps}", flush=True)
    print("Extracting frames (1 per second)...", flush=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % fps == 0:  # 1 frame per second
            timestamp = frame_count // fps

            # Convert to PIL and resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Resize to 768px max
            max_size = 768
            if pil_image.width > max_size or pil_image.height > max_size:
                if pil_image.width > pil_image.height:
                    new_width = max_size
                    new_height = int((max_size / pil_image.width) * pil_image.height)
                else:
                    new_height = max_size
                    new_width = int((max_size / pil_image.height) * pil_image.width)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save thumbnail
            thumbnail_path = os.path.join(thumbnails_folder, f"frame_{timestamp}.jpg")
            pil_image.save(thumbnail_path, format="JPEG", quality=75)

            # Encode to base64
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG", quality=75)
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            frames_to_process.append((timestamp, img_base64, thumbnail_path))

        frame_count += 1

    cap.release()

    total_frames = len(frames_to_process)
    if total_frames == 0:
        raise Exception("No frames extracted from video")
    
    print(f"Extracted {total_frames} frames to analyze", flush=True)
    print("Starting OpenAI Vision API analysis...", flush=True)

    # Process frames in batches with higher concurrency
    # Using semaphore to control max concurrent requests (avoid rate limits)
    # Can be configured via ANALYZE_CONCURRENT_REQUESTS env var (default: 20)
    max_concurrent = int(os.getenv("ANALYZE_CONCURRENT_REQUESTS", "20"))
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def analyze_with_semaphore(timestamp, img_base64):
        """Wrapper to limit concurrent API calls"""
        async with semaphore:
            return await analyze_frame_async(client, img_base64, timestamp)
    
    metadata = []
    batch_size = 50  # Larger batches for better throughput

    print(f"Processing {total_frames} frames with {max_concurrent} concurrent requests per batch...", flush=True)
    
    for i in range(0, len(frames_to_process), batch_size):
        batch = frames_to_process[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(frames_to_process) + batch_size - 1) // batch_size
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} frames)...", flush=True)

        # Process batch concurrently with semaphore limiting
        tasks = [analyze_with_semaphore(ts, img_base64) for ts, img_base64, _ in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for (timestamp, _, thumbnail_path), result in zip(batch, results):
            if isinstance(result, Exception):
                print(f"Error at {timestamp}s: {result}", flush=True)
                continue

            # Result is a tuple: (frame_data, error)
            frame_data, error = result
            
            if error:
                print(f"Frame {timestamp}s error: {error}", flush=True)
                continue

            # Add thumbnail path
            frame_data["thumbnail_path"] = thumbnail_path
            metadata.append(frame_data)

            # Report progress
            progress = int((len(metadata) / total_frames) * 100)
            if progress_callback:
                progress_callback(progress, timestamp)

            print(f"✓ Frame {timestamp}s analyzed ({len(metadata)}/{total_frames})", flush=True)

        # Small delay between batches to avoid overwhelming the API
        if i + batch_size < len(frames_to_process):
            await asyncio.sleep(0.1)  # Reduced delay since we have semaphore limiting

    # Save metadata
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Update cache
    update_video_entry(video_id, metadata_file=metadata_path, mtime=os.path.getmtime(video_path))

    print(f"✅ Analysis complete: {len(metadata)} frames saved to {metadata_path}", flush=True)
    return len(metadata)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_single_video.py <video_path>", flush=True)
        sys.exit(1)

    video_path = sys.argv[1]
    print(f"Starting analysis of: {video_path}", flush=True)
    
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}", flush=True)
        sys.exit(1)
    
    video_id = video_id_for_path(video_path)
    print(f"Video ID: {video_id}", flush=True)

    # Setup paths
    metadata_folder = os.path.join("data", "metadata")
    thumbnails_folder = os.path.join("data", "thumbnails", video_id)
    os.makedirs(metadata_folder, exist_ok=True)
    os.makedirs(thumbnails_folder, exist_ok=True)

    metadata_path = os.path.join(metadata_folder, f"{video_id}_objects.json")
    print(f"Metadata will be saved to: {metadata_path}", flush=True)

    # Run analysis
    try:
        print("Initializing OpenAI client...", flush=True)
        result = asyncio.run(process_video(video_path, video_id, metadata_path, thumbnails_folder))
        print(f"SUCCESS: Analyzed {result} frames", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

