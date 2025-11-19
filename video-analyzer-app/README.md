# Video Analyzer - Electron + React App

A beautiful, Premiere Pro-inspired desktop application for video analysis and semantic search powered by OpenAI Vision API.

## Features

🎨 **Premiere Pro-Inspired UI**
- Dark theme with professional color palette
- Timeline-style thumbnail grid
- Smooth animations and transitions
- Modern, responsive design

🔍 **Powerful Search**
- Semantic search powered by TF-IDF
- Search by vibes, moods, objects, locations
- Real-time filtering and results
- Tag-based organization

🎬 **Video Analysis**
- Frame-by-frame OpenAI Vision API analysis
- 20+ metadata fields per frame
- Cached thumbnails for instant loading
- Parallel processing for speed

📊 **Rich Metadata**
- Descriptions, moods, lighting, weather
- Objects, people, animals detection
- Geographic context and landmarks
- Time of day, camera angles, scene types

## Architecture

```
video-analyzer-app/          # Electron + React frontend
├── src/
│   ├── App.jsx              # Main React component
│   ├── App.css              # Premiere Pro-inspired styles
│   ├── api/
│   │   └── videoService.js  # API client for Python backend
│   └── main.jsx             # React entry point
├── electron/
│   ├── main.js              # Electron main process
│   └── preload.js           # Preload script
└── package.json

../src/                       # Python backend (existing)
├── api_server.py            # Flask REST API server
├── runMe_async.py           # Video analysis with OpenAI
└── utils/
    └── video_cache.py       # Video metadata cache
```

## Tech Stack

### Frontend
- **Electron** - Desktop app framework
- **React** - UI library
- **Vite** - Fast build tool
- **Lucide React** - Beautiful icons
- **Axios** - HTTP client

### Backend
- **Python 3.x** - Core language
- **Flask** - REST API server
- **OpenAI API** - Vision analysis
- **scikit-learn** - TF-IDF search
- **OpenCV** - Video processing

## Getting Started

### Prerequisites
- Node.js 16+
- Python 3.x
- OpenAI API key

### Installation

1. **Set up Python backend** (if not already done):
```bash
cd /path/to/video-checker
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install flask flask-cors
```

2. **Install frontend dependencies**:
```bash
cd video-analyzer-app
npm install
```

3. **Configure OpenAI API key**:
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your-api-key-here
```

### Running the App

**Option 1: Run Electron app (starts both frontend and backend)**:
```bash
cd video-analyzer-app
npm run electron
```

**Option 2: Run separately (for development)**:

Terminal 1 - Python backend:
```bash
source .venv/bin/activate
python src/api_server.py
```

Terminal 2 - Electron frontend:
```bash
cd video-analyzer-app
npm run electron
```

### Building for Production

```bash
cd video-analyzer-app
npm run electron:build
```

## API Endpoints

The Flask backend exposes the following REST API:

- `GET /api/health` - Health check
- `GET /api/videos` - List all videos
- `GET /api/videos/:id` - Get video details
- `GET /api/videos/:id/metadata` - Get frame analysis
- `POST /api/search` - Search videos (body: `{query, filters}`)
- `POST /api/videos/register` - Register new videos (body: `{paths}`)

## Development

### Project Structure

**Frontend Components**:
- `App.jsx` - Main app with sidebar, search, thumbnail grid
- `videoService.js` - API client for backend communication
- `App.css` - Premiere Pro-inspired design system

**Key Features**:
- Dark theme with `#1a1a1a` primary background
- Accent colors: Blue (`#00a4ff`), Purple (`#8b5cf6`), Green (`#10b981`)
- Grid layout with 280px thumbnails
- Smooth hover animations and transitions

### Customization

**Colors** (in `App.css`):
```css
:root {
  --bg-primary: #1a1a1a;
  --accent-blue: #00a4ff;
  --text-primary: #e5e5e5;
  /* ... */
}
```

**Grid Layout**:
```css
.thumbnail-grid {
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
}
```

## Workflow

1. **Upload Videos**: Select video files to analyze
2. **Analysis**: Backend extracts frames → OpenAI API → Metadata saved
3. **Search**: Type queries like "peaceful nature", "dog running", "sunset"
4. **Browse**: View results in grid or list mode
5. **Open**: Click thumbnails to jump to exact timestamps

## Performance

- **Parallel Processing**: 5 frames analyzed concurrently
- **Cached Thumbnails**: Instant loading from disk
- **Smart Frame Extraction**: 1 frame per second
- **Optimized Images**: 768px max, 75% JPEG quality

## Troubleshooting

**"Backend API is not running"**:
- Ensure Python Flask server is running on port 5000
- Check `python src/api_server.py` is running

**Thumbnails not showing**:
- Verify `data/thumbnails/{video_id}/` folders exist
- Check file paths in metadata JSON files

**Search returns no results**:
- Ensure videos have been analyzed (check `data/metadata/`)
- Try broader search terms

## Future Enhancements

- [ ] Video preview on thumbnail hover
- [ ] Timeline scrubbing
- [ ] Export search results
- [ ] Batch video upload with drag-and-drop
- [ ] Advanced filters (date, duration, resolution)
- [ ] Video player integration
- [ ] Clip extraction and export

## License

MIT

