import sys
import os
import json
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QTabWidget, QListWidget, QLineEdit, QListWidgetItem, QTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import cv2

from utils.video_cache import (
    get_videos_to_process,
    register_videos,
    video_id_for_path,
    get_video_entry,
    iter_registered_videos,
)


class VideoAnalysisThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, videos_to_process):
        super().__init__()
        self.videos_to_process = videos_to_process

    def run(self):
        try:
            import base64
            from io import BytesIO
            from PIL import Image
            from openai import AsyncOpenAI
            from dotenv import load_dotenv
            from utils.video_cache import update_video_entry
            import asyncio

            load_dotenv()
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            metadata_folder = os.path.join("data", "metadata")
            os.makedirs(metadata_folder, exist_ok=True)

            self.progress.emit(f"Starting analysis of {len(self.videos_to_process)} video(s)...\n")

            # Run async processing
            asyncio.run(self.process_videos_async(client, metadata_folder))

            self.progress.emit("\n✅ Analysis complete!\n")
            self.finished.emit()

        except Exception as e:
            self.progress.emit(f"\n❌ Error during analysis: {e}\n")
            self.finished.emit()

    async def analyze_frame_async(self, client, img_base64, timestamp, max_retries=5):
        """Analyze a single frame with retry logic"""
        import base64
        import time

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
12. **People**: List all people visible and what they're doing (e.g., ["person walking", "child playing", "group talking"]).
13. **Animals**: List all animals visible (e.g., ["dog running", "bird flying", "cat sitting"]).
14. **Objects**: List ALL visible objects in detail - be exhaustive. Include furniture, vehicles, natural elements, buildings, tools, food, clothing items, electronics, etc. Return as a comprehensive list of strings.
15. **Actions/Activities**: List all activities or actions occurring (e.g., ["walking", "talking", "eating", "driving"]).
16. **Text Visible**: Any visible text, signs, or labels in the frame (list of strings, or empty list if none).
17. **Location Type**: Type of location ("Indoor", "Outdoor", "Studio", "Urban", "Rural", "Natural", "Industrial", "Residential", "Commercial").
18. **Scene Type**: Category of scene ("Landscape", "Portrait", "Action", "Dialogue", "Establishing Shot", "B-Roll", "Product Shot", "Close-up Detail").

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
  "scene_type": "string"
}"""

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

            except Exception as api_error:
                error_message = str(api_error)
                if "rate_limit" in error_message.lower() or "429" in error_message:
                    if attempt < max_retries - 1:
                        wait_time = 2 * (2 ** attempt)
                        await asyncio.sleep(wait_time)
                        continue
                return None, str(api_error)

        return None, "Max retries reached"

    async def process_videos_async(self, client, metadata_folder):
        """Process videos with concurrent frame analysis"""
        import cv2
        import base64
        from io import BytesIO
        from PIL import Image
        from utils.video_cache import update_video_entry

        for idx, video_entry in enumerate(self.videos_to_process, start=1):
            video_path = video_entry.get("path")
            video_id = video_entry.get("id")
            filename = video_entry.get("filename") or os.path.basename(video_path)
            metadata_path = video_entry.get("metadata_file") or os.path.join(
                metadata_folder, f"{video_id}_objects.json"
            )

            self.progress.emit(f"\n[{idx}/{len(self.videos_to_process)}] Processing: {filename}\n")

            if not os.path.exists(video_path):
                self.progress.emit(f"  ⚠ Video not found: {video_path}\n")
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.progress.emit(f"  ⚠ Could not open video: {filename}\n")
                continue

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0:
                self.progress.emit(f"  ⚠ Invalid FPS for: {filename}\n")
                cap.release()
                continue

            # Extract all frames first
            frame_data_list = []
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process one frame per second
                if frame_count % fps == 0:
                        timestamp = frame_count // fps

                        # Convert frame to base64 for OpenAI with reduced resolution
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)

                        # Resize to max 768px on longest side to reduce token usage and avoid rate limits
                        max_size = 768
                        if pil_image.width > max_size or pil_image.height > max_size:
                            if pil_image.width > pil_image.height:
                                new_width = max_size
                                new_height = int((max_size / pil_image.width) * pil_image.height)
                            else:
                                new_height = max_size
                                new_width = int((max_size / pil_image.height) * pil_image.width)
                            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

                        buffered = BytesIO()
                        pil_image.save(buffered, format="JPEG", quality=75)
                        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                        self.progress.emit(f"  Analyzing frame at {timestamp}s...\n")

                        # Retry logic for rate limiting
                        max_retries = 5
                        retry_delay = 2

                        for attempt in range(max_retries):
                            try:
                                # Call OpenAI Vision API with comprehensive prompt
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
12. **People**: List all people visible and what they're doing (e.g., ["person walking", "child playing", "group talking"]).
13. **Animals**: List all animals visible (e.g., ["dog running", "bird flying", "cat sitting"]).
14. **Objects**: List ALL visible objects in detail - be exhaustive. Include furniture, vehicles, natural elements, buildings, tools, food, clothing items, electronics, etc. Return as a comprehensive list of strings.
15. **Actions/Activities**: List all activities or actions occurring (e.g., ["walking", "talking", "eating", "driving"]).
16. **Text Visible**: Any visible text, signs, or labels in the frame (list of strings, or empty list if none).
17. **Location Type**: Type of location ("Indoor", "Outdoor", "Studio", "Urban", "Rural", "Natural", "Industrial", "Residential", "Commercial").
18. **Scene Type**: Category of scene ("Landscape", "Portrait", "Action", "Dialogue", "Establishing Shot", "B-Roll", "Product Shot", "Close-up Detail").

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
  "scene_type": "string"
}"""

                                response = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[
                                        {
                                            "role": "user",
                                            "content": [
                                                {
                                                    "type": "text",
                                                    "text": prompt
                                                },
                                                {
                                                    "type": "image_url",
                                                    "image_url": {
                                                        "url": f"data:image/jpeg;base64,{img_base64}"
                                                    }
                                                }
                                            ]
                                        }
                                    ],
                                    max_tokens=1500,
                                    response_format={"type": "json_object"}
                                )

                                # Parse response
                                response_text = response.choices[0].message.content
                                frame_data = json.loads(response_text)

                                frame_data["timestamp"] = timestamp
                                metadata.append(frame_data)

                                # Display summary
                                objects = frame_data.get("objects", [])
                                main_subject = frame_data.get("main_subject", "N/A")
                                setting = frame_data.get("setting", "N/A")
                                time_of_day = frame_data.get("time_of_day", "N/A")
                                camera_angle = frame_data.get("camera_angle", "N/A")

                                if objects:
                                    obj_count = len(objects)
                                    obj_str = ", ".join(objects[:8])
                                    if obj_count > 8:
                                        obj_str += f"... (+{obj_count - 8} more)"
                                    self.progress.emit(f"    ✓ {obj_count} Objects: {obj_str}\n")
                                self.progress.emit(f"    Subject: {main_subject} | Setting: {setting}\n")
                                self.progress.emit(f"    Time: {time_of_day} | Angle: {camera_angle}\n")

                                break  # Success, exit retry loop

                            except Exception as api_error:
                                error_message = str(api_error)

                                # Check if it's a rate limit error
                                if "rate_limit" in error_message.lower() or "429" in error_message:
                                    if attempt < max_retries - 1:
                                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                                        self.progress.emit(f"    ⚠ Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...\n")
                                        import time
                                        time.sleep(wait_time)
                                        continue
                                    else:
                                        self.progress.emit(f"    ❌ Max retries reached at {timestamp}s: {api_error}\n")
                                        metadata.append({
                                            "timestamp": timestamp,
                                            "error": "rate_limit_exceeded",
                                            "objects": [],
                                            "description": "Rate limit exceeded"
                                        })
                                        break
                                else:
                                    self.progress.emit(f"    ⚠ API error at {timestamp}s: {api_error}\n")
                                    metadata.append({
                                        "timestamp": timestamp,
                                        "error": str(api_error),
                                        "objects": [],
                                        "description": "error"
                                    })
                                    break

                    frame_count += 1

                cap.release()

                with open(metadata_path, "w", encoding="utf-8") as json_file:
                    json.dump(metadata, json_file, indent=4)

                update_video_entry(video_id, metadata_file=metadata_path, mtime=os.path.getmtime(video_path))
                self.progress.emit(f"  ✓ Saved metadata to: {metadata_path}\n")

            self.progress.emit("\n✅ Analysis complete!\n")
            self.finished.emit()

        except Exception as e:
            self.progress.emit(f"\n❌ Error during analysis: {e}\n")
            self.finished.emit()


class VideoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.videoPaths = []

    def initUI(self):
        self.setWindowTitle("Video Analyzer")
        self.resize(900, 700)

        # Main layout
        layout = QVBoxLayout()

        # Tab widget
        self.tabs = QTabWidget()
        self.uploadTab = QWidget()
        self.searchTab = QWidget()

        self.tabs.addTab(self.uploadTab, "Upload Videos")
        self.tabs.addTab(self.searchTab, "Search Videos")

        self.initUploadTab()
        self.initSearchTab()

        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def initUploadTab(self):
        layout = QVBoxLayout()

        # Upload button
        self.uploadButton = QPushButton("Select Videos")
        self.uploadButton.clicked.connect(self.uploadVideos)
        layout.addWidget(self.uploadButton)

        # Status label
        self.uploadStatusLabel = QLabel("")
        self.uploadStatusLabel.setWordWrap(True)
        layout.addWidget(self.uploadStatusLabel)

        # Root info label
        self.rootInfoLabel = QLabel("")
        self.rootInfoLabel.setWordWrap(True)
        self.rootInfoLabel.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(self.rootInfoLabel)

        # Open root button
        self.openRootButton = QPushButton("Open Video Root Folder")
        self.openRootButton.setEnabled(False)
        self.openRootButton.clicked.connect(self.openVideoRoot)
        layout.addWidget(self.openRootButton)

        # Analyze button
        self.analyzeButton = QPushButton("Analyze Videos")
        self.analyzeButton.setEnabled(False)
        self.analyzeButton.clicked.connect(self.analyzeVideos)
        layout.addWidget(self.analyzeButton)

        # Analysis progress text area (streaming output)
        self.analysisProgressLabel = QLabel("Analysis progress:")
        layout.addWidget(self.analysisProgressLabel)

        self.analysisProgressText = QTextEdit()
        self.analysisProgressText.setReadOnly(True)
        self.analysisProgressText.setMaximumHeight(300)
        layout.addWidget(self.analysisProgressText)

        layout.addStretch()
        self.uploadTab.setLayout(layout)

    def initSearchTab(self):
        layout = QVBoxLayout()

        # Search bar
        searchLayout = QHBoxLayout()
        self.searchInput = QLineEdit()
        self.searchInput.setPlaceholderText("Search for objects (e.g., person, car, dog)...")
        self.searchButton = QPushButton("Search")
        self.searchButton.clicked.connect(self.searchVideos)
        searchLayout.addWidget(self.searchInput)
        searchLayout.addWidget(self.searchButton)
        layout.addLayout(searchLayout)

        # Results list
        self.resultsLabel = QLabel("Search results will appear below:")
        layout.addWidget(self.resultsLabel)

        self.resultsList = QListWidget()
        self.resultsList.itemDoubleClicked.connect(self.openVideoAtTimestamp)
        layout.addWidget(self.resultsList)

        self.searchTab.setLayout(layout)

    def uploadVideos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", "", "Video Files (*.mp4 *.mov *.mkv *.avi)"
        )
        if not files:
            self.uploadStatusLabel.setText("No videos selected.")
            self.rootInfoLabel.setText("")
            self.openRootButton.setEnabled(False)
            self.analyzeButton.setEnabled(False)
            return

        self.videoPaths = [os.path.abspath(path) for path in files]
        register_videos(self.videoPaths)

        unique_roots = sorted({os.path.dirname(path) for path in self.videoPaths})
        root_lines = "\n".join(unique_roots)
        self.rootInfoLabel.setText(f"Video root folders:\n{root_lines}")

        videos_to_process, _ = get_videos_to_process(self.videoPaths)
        if videos_to_process:
            self.uploadStatusLabel.setText(
                f"Selected {len(files)} video(s). {len(videos_to_process)} need analysis."
            )
        else:
            self.uploadStatusLabel.setText(
                f"Selected {len(files)} video(s). All are already analyzed."
            )

        self.openRootButton.setEnabled(True)
        self.analyzeButton.setEnabled(True)

    def openVideoRoot(self):
        if not self.videoPaths:
            return

        root_path = os.path.dirname(self.videoPaths[0])
        if not root_path or not os.path.exists(root_path):
            self.uploadStatusLabel.setText("Root folder not available.")
            return

        try:
            if sys.platform.startswith("darwin"):
                subprocess.run(["open", root_path], check=False)
            elif os.name == "nt" and hasattr(os, "startfile"):
                os.startfile(root_path)
            else:
                subprocess.run(["xdg-open", root_path], check=False)
        except Exception as exc:
            self.uploadStatusLabel.setText(f"Unable to open folder: {exc}")

    def analyzeVideos(self):
        if not self.videoPaths:
            self.analysisProgressText.append("No videos selected.")
            return

        videos_to_process, _ = get_videos_to_process(self.videoPaths)

        if not videos_to_process:
            self.analysisProgressText.append("All videos are already analyzed.")
            return

        self.analyzeButton.setEnabled(False)
        self.analysisProgressText.clear()
        self.analysisProgressText.append("Starting analysis...\n")

        self.analysisThread = VideoAnalysisThread(videos_to_process)
        self.analysisThread.progress.connect(self.updateAnalysisProgress)
        self.analysisThread.finished.connect(self.onAnalysisFinished)
        self.analysisThread.start()

    def updateAnalysisProgress(self, message):
        self.analysisProgressText.insertPlainText(message)
        self.analysisProgressText.ensureCursorVisible()

    def onAnalysisFinished(self):
        self.analyzeButton.setEnabled(True)

    def searchVideos(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        search_query = self.searchInput.text().strip()
        if not search_query:
            self.resultsLabel.setText("Please enter a search term or describe what you're looking for (e.g., 'sad scene', 'peaceful nature').")
            return

        self.resultsList.clear()
        self.resultsLabel.setText("Searching with semantic matching...")

        # Collect all frame data with rich text descriptions
        all_frames = []

        for video_id, entry in iter_registered_videos():
            metadata_file = entry.get("metadata_file")
            if not metadata_file or not os.path.exists(metadata_file):
                continue

            video_path = entry.get("path")
            filename = entry.get("filename") or video_id

            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                for frame_data in metadata:
                    timestamp = frame_data.get("timestamp", 0)

                    # Build comprehensive searchable text for semantic matching
                    description = frame_data.get("description", "")
                    main_subject = frame_data.get("main_subject", "")
                    setting = frame_data.get("setting", "")
                    time_of_day = frame_data.get("time_of_day", "")
                    lighting = frame_data.get("lighting", "")
                    weather = frame_data.get("weather", "")
                    motion = frame_data.get("motion", "")
                    color_palette = frame_data.get("color_palette", "")
                    mood = frame_data.get("mood", "")
                    people = frame_data.get("people", [])
                    animals = frame_data.get("animals", [])
                    objects = frame_data.get("objects", [])
                    actions = frame_data.get("actions", [])
                    location_type = frame_data.get("location_type", "")
                    scene_type = frame_data.get("scene_type", "")

                    # Combine all fields into rich searchable text
                    searchable_text = " ".join([
                        description,
                        main_subject,
                        setting,
                        time_of_day,
                        lighting,
                        weather,
                        motion,
                        color_palette,
                        mood,
                        " ".join(people),
                        " ".join(animals),
                        " ".join(objects),
                        " ".join(actions),
                        location_type,
                        scene_type,
                    ]).strip()

                    all_frames.append({
                        "video_path": video_path,
                        "filename": filename,
                        "timestamp": timestamp,
                        "text": searchable_text,
                        "description": description,
                        "mood": mood,
                        "setting": setting,
                        "main_subject": main_subject,
                    })

            except (json.JSONDecodeError, OSError) as exc:
                print(f"Error reading metadata for {filename}: {exc}")

        # Perform semantic search using TF-IDF vectorization
        if not all_frames:
            self.resultsLabel.setText("No analyzed videos found. Please analyze videos first.")
            return

        texts = [frame["text"] for frame in all_frames]

        try:
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),
                stop_words='english',
                lowercase=True
            )

            # Fit and transform all frame texts
            tfidf_matrix = vectorizer.fit_transform(texts)

            # Transform the search query
            query_vector = vectorizer.transform([search_query])

            # Compute cosine similarity
            similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

            # Get top matches (above threshold)
            threshold = 0.05
            top_indices = np.argsort(similarities)[::-1]

            results_found = False
            for idx in top_indices:
                similarity = similarities[idx]
                if similarity < threshold:
                    break

                frame = all_frames[idx]
                score_pct = int(similarity * 100)

                item_text = f"[{score_pct}%] {frame['filename']} - {frame['description'][:70]} at {frame['timestamp']}s"
                if frame['mood']:
                    item_text += f" | Mood: {frame['mood']}"

                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, (frame['video_path'], frame['timestamp']))
                self.resultsList.addItem(item)
                results_found = True

            if results_found:
                self.resultsLabel.setText(f"Found {self.resultsList.count()} semantic matches for '{search_query}':")
            else:
                self.resultsLabel.setText(f"No strong matches found for '{search_query}'. Try different words or descriptions.")

        except Exception as e:
            self.resultsLabel.setText(f"Search error: {e}")
            print(f"Search error: {e}")

            # Fallback to simple keyword search
            search_term = search_query.lower()
            for frame in all_frames:
                if search_term in frame["text"].lower():
                    item_text = f"{frame['filename']} - {frame['description'][:70]} at {frame['timestamp']}s"
                    item = QListWidgetItem(item_text)
                    item.setData(Qt.UserRole, (frame['video_path'], frame['timestamp']))
                    self.resultsList.addItem(item)

    def openVideoAtTimestamp(self, item):
        video_path, timestamp = item.data(Qt.UserRole)

        if not video_path or not os.path.exists(video_path):
            self.resultsLabel.setText("Video file not found.")
            return

        try:
            if sys.platform.startswith("darwin"):
                subprocess.run(["open", video_path], check=False)
            elif os.name == "nt" and hasattr(os, "startfile"):
                os.startfile(video_path)
            else:
                subprocess.run(["xdg-open", video_path], check=False)
        except Exception as exc:
            self.resultsLabel.setText(f"Unable to open video: {exc}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoApp()
    window.show()
    sys.exit(app.exec_())
