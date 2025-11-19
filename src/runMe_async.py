import sys
import os
import json
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QTabWidget, QListWidget, QLineEdit, QListWidgetItem, QTextEdit,
    QScrollArea, QGridLayout, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage
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

            # Create thumbnails folder
            thumbnails_folder = os.path.join("data", "thumbnails")
            os.makedirs(thumbnails_folder, exist_ok=True)

            self.progress.emit(f"Starting analysis of {len(self.videos_to_process)} video(s)...\n")

            # Run async processing
            asyncio.run(self.process_videos_async(client, metadata_folder, thumbnails_folder))

            self.progress.emit("\n✅ Analysis complete!\n")
            self.finished.emit()

        except Exception as e:
            self.progress.emit(f"\n❌ Error during analysis: {e}\n")
            self.finished.emit()

    async def analyze_frame_async(self, client, img_base64, timestamp):
        """Analyze a single frame asynchronously"""
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
            return None, str(e)

    async def process_videos_async(self, client, metadata_folder, thumbnails_folder):
        """Process videos with concurrent frame analysis (5 frames at a time)"""
        import cv2
        import base64
        from io import BytesIO
        from PIL import Image
        from utils.video_cache import update_video_entry
        import asyncio

        for idx, video_entry in enumerate(self.videos_to_process, start=1):
            video_path = video_entry.get("path")
            video_id = video_entry.get("id")
            filename = video_entry.get("filename") or os.path.basename(video_path)
            metadata_path = video_entry.get("metadata_file") or os.path.join(
                metadata_folder, f"{video_id}_objects.json"
            )

            # Create video-specific thumbnail folder
            video_thumbnails_folder = os.path.join(thumbnails_folder, video_id)
            os.makedirs(video_thumbnails_folder, exist_ok=True)

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

            # Extract all frames into memory first
            frames_to_process = []
            frame_count = 0

            self.progress.emit(f"  Extracting frames and saving thumbnails...\n")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % fps == 0:
                    timestamp = frame_count // fps

                    # Convert to base64
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

                    # Save thumbnail to disk
                    thumbnail_path = os.path.join(video_thumbnails_folder, f"frame_{timestamp}.jpg")
                    pil_image.save(thumbnail_path, format="JPEG", quality=75)

                    buffered = BytesIO()
                    pil_image.save(buffered, format="JPEG", quality=75)
                    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                    frames_to_process.append((timestamp, img_base64, thumbnail_path))

                frame_count += 1

            cap.release()

            self.progress.emit(f"  Analyzing {len(frames_to_process)} frames with parallel processing (batch size: 5)...\n")

            # Process frames in batches of 5 concurrently
            metadata = []
            batch_size = 5

            for i in range(0, len(frames_to_process), batch_size):
                batch = frames_to_process[i:i + batch_size]

                # Process batch concurrently
                tasks = [self.analyze_frame_async(client, img_base64, ts) for ts, img_base64, _ in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for (timestamp, _, thumbnail_path), result in zip(batch, results):
                    if isinstance(result, Exception):
                        self.progress.emit(f"    ❌ Error at {timestamp}s: {result}\n")
                        continue

                    frame_data, error = result
                    if error:
                        self.progress.emit(f"    ⚠ Frame {timestamp}s: {error}\n")
                        continue

                    # Add thumbnail path to frame data
                    frame_data["thumbnail_path"] = thumbnail_path
                    metadata.append(frame_data)

                    objects = frame_data.get("objects", [])
                    if objects:
                        obj_count = len(objects)
                        obj_str = ", ".join(objects[:5])
                        if obj_count > 5:
                            obj_str += f"... (+{obj_count - 5} more)"
                        self.progress.emit(f"    ✓ Frame {timestamp}s: {obj_count} objects - {obj_str}\n")
                    else:
                        self.progress.emit(f"    ✓ Frame {timestamp}s analyzed\n")

                # Small delay between batches to avoid overwhelming the API
                if i + batch_size < len(frames_to_process):
                    await asyncio.sleep(0.5)

            # Save metadata
            with open(metadata_path, "w", encoding="utf-8") as json_file:
                json.dump(metadata, json_file, indent=4)

            update_video_entry(video_id, metadata_file=metadata_path, mtime=os.path.getmtime(video_path))
            self.progress.emit(f"  ✓ Saved metadata to: {metadata_path}\n")


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
        self.analyzeButton = QPushButton("Analyze Videos (Parallel Processing)")
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

        # Search bar and controls
        searchLayout = QHBoxLayout()
        self.searchInput = QLineEdit()
        self.searchInput.setPlaceholderText("Search for vibes, objects, moods (e.g., 'sad scene', 'dog', 'sunset')...")
        self.searchInput.returnPressed.connect(self.searchVideos)  # Search on Enter
        self.searchButton = QPushButton("Search")
        self.searchButton.clicked.connect(self.searchVideos)
        self.clearButton = QPushButton("Clear")
        self.clearButton.clicked.connect(self.showTopResults)

        # View toggle button
        self.viewToggleButton = QPushButton("📷 Thumbnail View")
        self.viewToggleButton.clicked.connect(self.toggleView)
        self.currentView = "list"  # Start with list view

        searchLayout.addWidget(self.searchInput)
        searchLayout.addWidget(self.searchButton)
        searchLayout.addWidget(self.clearButton)
        searchLayout.addWidget(self.viewToggleButton)
        layout.addLayout(searchLayout)

        # Results label
        self.resultsLabel = QLabel("Loading top results from analyzed videos...")
        layout.addWidget(self.resultsLabel)

        # List view
        self.resultsList = QListWidget()
        self.resultsList.itemDoubleClicked.connect(self.openVideoAtTimestamp)
        layout.addWidget(self.resultsList)

        # Thumbnail view (scroll area with grid)
        self.thumbnailScrollArea = QScrollArea()
        self.thumbnailScrollArea.setWidgetResizable(True)
        self.thumbnailScrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.thumbnailWidget = QWidget()
        self.thumbnailLayout = QGridLayout()
        self.thumbnailLayout.setSpacing(10)
        self.thumbnailWidget.setLayout(self.thumbnailLayout)
        self.thumbnailScrollArea.setWidget(self.thumbnailWidget)
        self.thumbnailScrollArea.hide()  # Start hidden

        layout.addWidget(self.thumbnailScrollArea)

        self.searchTab.setLayout(layout)

        # Store current results for view switching
        self.currentResults = []

        # Load top results on init (when tab is first shown)
        self.tabs.currentChanged.connect(self.onTabChanged)

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
                f"Selected {len(files)} video(s). {len(videos_to_process)} need analysis (5 frames at a time!)."
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
        self.analysisProgressText.append("Starting parallel analysis (5 frames at a time)...\n")

        self.analysisThread = VideoAnalysisThread(videos_to_process)
        self.analysisThread.progress.connect(self.updateAnalysisProgress)
        self.analysisThread.finished.connect(self.onAnalysisFinished)
        self.analysisThread.start()

    def updateAnalysisProgress(self, message):
        self.analysisProgressText.insertPlainText(message)
        self.analysisProgressText.ensureCursorVisible()

    def onAnalysisFinished(self):
        self.analyzeButton.setEnabled(True)

    def onTabChanged(self, index):
        """Load top results when switching to search tab"""
        if index == 1 and self.resultsList.count() == 0:  # Search tab index is 1
            self.showTopResults()

    def toggleView(self):
        """Toggle between list and thumbnail view"""
        if self.currentView == "list":
            # Switch to thumbnail view
            self.currentView = "thumbnail"
            self.viewToggleButton.setText("📝 List View")
            self.resultsList.hide()
            self.thumbnailScrollArea.show()
            self.displayThumbnailView()
        else:
            # Switch to list view
            self.currentView = "list"
            self.viewToggleButton.setText("📷 Thumbnail View")
            self.thumbnailScrollArea.hide()
            self.resultsList.show()

    def loadThumbnailFromCache(self, thumbnail_path):
        """Load a cached thumbnail from disk"""
        try:
            if thumbnail_path and os.path.exists(thumbnail_path):
                pixmap = QPixmap(thumbnail_path)
                if not pixmap.isNull():
                    return pixmap
        except Exception as e:
            print(f"Error loading cached thumbnail: {e}")
        return None

    def extractFrameAtTimestamp(self, video_path, timestamp):
        """Extract a single frame from video at given timestamp (fallback if no cache)"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None

            # Seek to timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            cap.release()

            if ret:
                return frame
            return None
        except Exception as e:
            print(f"Error extracting frame: {e}")
            return None

    def createThumbnailCard(self, frame_info, row, col):
        """Create a thumbnail card widget for grid display"""
        card = QFrame()
        card.setFrameStyle(QFrame.Box | QFrame.Raised)
        card.setLineWidth(2)
        card.setMaximumWidth(300)
        card.setStyleSheet("""
            QFrame {
                background-color: #f5f5f5;
                border: 2px solid #ddd;
                border-radius: 8px;
                padding: 5px;
            }
            QFrame:hover {
                border: 2px solid #4CAF50;
                background-color: #e8f5e9;
            }
        """)
        card.setCursor(Qt.PointingHandCursor)

        cardLayout = QVBoxLayout()
        cardLayout.setSpacing(5)

        # Get thumbnail info
        video_path = frame_info.get("video_path")
        timestamp = frame_info.get("timestamp", 0)
        thumbnail_path = frame_info.get("thumbnail_path")

        # Create thumbnail image
        thumbnailLabel = QLabel()
        pixmap = None

        # Try to load from cache first
        if thumbnail_path:
            pixmap = self.loadThumbnailFromCache(thumbnail_path)

        # Fallback to extracting frame if no cache
        if pixmap is None:
            frame = self.extractFrameAtTimestamp(video_path, timestamp)
            if frame is not None:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)

        if pixmap is not None:
            # Scale to thumbnail size
            scaled_pixmap = pixmap.scaled(280, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            thumbnailLabel.setPixmap(scaled_pixmap)
            thumbnailLabel.setAlignment(Qt.AlignCenter)
        else:
            thumbnailLabel.setText("🎬 No Preview")
            thumbnailLabel.setAlignment(Qt.AlignCenter)
            thumbnailLabel.setStyleSheet("font-size: 24px; color: #999;")

        cardLayout.addWidget(thumbnailLabel)

        # Info text
        filename = frame_info.get("filename", "Unknown")
        description = frame_info.get("description", "")[:60] + "..."
        mood = frame_info.get("mood", "")
        landmarks = frame_info.get("landmarks", [])

        infoText = f"<b>{filename}</b><br>"
        infoText += f"⏱ {timestamp:.1f}s<br>"
        infoText += f"<small>{description}</small>"

        if mood:
            infoText += f"<br><small>😊 {mood}</small>"
        if landmarks:
            infoText += f"<br><small>📍 {', '.join(landmarks[:2])}</small>"

        infoLabel = QLabel(infoText)
        infoLabel.setWordWrap(True)
        infoLabel.setMaximumWidth(280)
        cardLayout.addWidget(infoLabel)

        card.setLayout(cardLayout)

        # Make card clickable
        card.mousePressEvent = lambda event: self.openVideoAtTimestampFromData(video_path, timestamp)

        return card

    def displayThumbnailView(self):
        """Display current results in thumbnail grid"""
        # Clear existing thumbnails
        while self.thumbnailLayout.count():
            item = self.thumbnailLayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self.currentResults:
            label = QLabel("No results to display")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-size: 16px; color: #999; padding: 50px;")
            self.thumbnailLayout.addWidget(label, 0, 0)
            return

        # Display results in grid (3 columns)
        cols = 3
        for idx, frame_info in enumerate(self.currentResults):
            row = idx // cols
            col = idx % cols
            card = self.createThumbnailCard(frame_info, row, col)
            self.thumbnailLayout.addWidget(card, row, col)

        # Add stretch to push cards to top
        self.thumbnailLayout.setRowStretch(len(self.currentResults) // cols + 1, 1)

    def openVideoAtTimestampFromData(self, video_path, timestamp):
        """Open video at specific timestamp (from data, not list item)"""
        if not video_path or not os.path.exists(video_path):
            return

        try:
            subprocess.Popen(
                ["open", "-a", "QuickTime Player", video_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"Error opening video: {e}")

    def showTopResults(self):
        """Show top interesting results from analyzed videos"""
        self.searchInput.clear()
        self.resultsList.clear()
        self.resultsLabel.setText("Loading top results...")

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
                    description = frame_data.get("description", "")
                    mood = frame_data.get("mood", "")
                    objects = frame_data.get("objects", [])
                    landmarks = frame_data.get("landmarks", [])
                    scene_type = frame_data.get("scene_type", "")
                    thumbnail_path = frame_data.get("thumbnail_path")

                    # Score frames based on "interestingness"
                    score = 0
                    if landmarks:
                        score += 50  # Landmarks are very interesting
                    if len(objects) > 10:
                        score += 20  # Rich scenes are interesting
                    if mood and mood.lower() not in ["n/a", "none", ""]:
                        score += 10  # Strong mood is interesting
                    if scene_type and scene_type.lower() in ["landscape", "action", "establishing shot"]:
                        score += 15  # Cinematic scenes
                    if len(description) > 100:
                        score += 10  # Detailed descriptions

                    all_frames.append({
                        "video_path": video_path,
                        "filename": filename,
                        "timestamp": timestamp,
                        "description": description,
                        "mood": mood,
                        "score": score,
                        "objects": objects,
                        "landmarks": landmarks,
                        "thumbnail_path": thumbnail_path,
                    })

            except (json.JSONDecodeError, OSError) as exc:
                print(f"Error reading metadata for {filename}: {exc}")

        if not all_frames:
            self.resultsLabel.setText("No analyzed videos found. Please analyze videos first in the Upload tab.")
            return

        # Sort by score and show top results
        all_frames.sort(key=lambda x: x["score"], reverse=True)
        top_results = all_frames[:50]  # Show top 50

        # Store current results for view switching
        self.currentResults = top_results

        for frame in top_results:
            item_text = f"⭐ {frame['filename']} - {frame['description'][:70]} at {frame['timestamp']}s"
            if frame['mood']:
                item_text += f" | {frame['mood']}"
            if frame['landmarks']:
                item_text += f" | 📍 {', '.join(frame['landmarks'][:2])}"

            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, (frame['video_path'], frame['timestamp']))
            self.resultsList.addItem(item)

        self.resultsLabel.setText(f"Top {len(top_results)} most interesting frames (landmarks, rich scenes, strong moods):")

        # If in thumbnail view, update thumbnails
        if self.currentView == "thumbnail":
            self.displayThumbnailView()

    def searchVideos(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        search_query = self.searchInput.text().strip()
        if not search_query:
            # If empty, show top results instead
            self.showTopResults()
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
                    thumbnail_path = frame_data.get("thumbnail_path")

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
                    geographic_context = frame_data.get("geographic_context", [])
                    landmarks = frame_data.get("landmarks", [])

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
                        " ".join(geographic_context),
                        " ".join(landmarks),
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
                        "thumbnail_path": thumbnail_path,
                        "landmarks": landmarks,
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
            search_results = []

            for idx in top_indices:
                similarity = similarities[idx]
                if similarity < threshold:
                    break

                frame = all_frames[idx]
                score_pct = int(similarity * 100)

                # Store for thumbnail view
                search_results.append(frame)

                item_text = f"[{score_pct}%] {frame['filename']} - {frame['description'][:70]} at {frame['timestamp']}s"
                if frame['mood']:
                    item_text += f" | Mood: {frame['mood']}"

                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, (frame['video_path'], frame['timestamp']))
                self.resultsList.addItem(item)
                results_found = True

            # Store current results for view switching
            self.currentResults = search_results

            if results_found:
                self.resultsLabel.setText(f"Found {self.resultsList.count()} semantic matches for '{search_query}':")
            else:
                self.resultsLabel.setText(f"No strong matches found for '{search_query}'. Try different words or descriptions.")

            # If in thumbnail view, update thumbnails
            if self.currentView == "thumbnail":
                self.displayThumbnailView()

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

