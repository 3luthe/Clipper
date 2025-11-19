# Clipper

Clipper is an AI-assisted video editor that stitches clips together around a generated narration. It was originally built for Carnegie Mellon's Tartan Hacks 2025 (sponsored by AppLovin).

## Repository Layout

- `src/` – Python pipeline for generating narration, matching scenes, and exporting a final edit.
- `voiceover_app/` – React + Flask prototype for experimenting with Amazon Polly voices.
- `data/` – Sample metadata, scripts, and audio used by the pipeline.
- `outputs/` – Example export created by the pipeline.

## Prerequisites

- Python 3.10+
- Node.js 18+ (for `voiceover_app`)
- FFmpeg (required by `moviepy` for video rendering)

## Environment Variables

Copy `env.sample` to `.env` and fill in your credentials:

```
cp env.sample .env
```

| Variable | Description |
| --- | --- |
| `OPENAI_API_KEY` | Used for GPT-driven metadata generation and narration |
| `ELEVENLABS_API_KEY` | Used to synthesize the narration audio |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | Optional. Required for Amazon Polly experiments in `voiceover_app` and `src/polly.py` |

The Python scripts load variables with `python-dotenv`, so storing them in `.env` is sufficient for local runs.

## Python Setup

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
nltk-downloader punkt
```

> **Note**: Installing `torch` and `ultralytics` can take several minutes. The default CPU wheels are requested via `pip`, but you can point to a different index if you need GPU support.

Ensure FFmpeg is on your `PATH`. On macOS you can install it via Homebrew: `brew install ffmpeg`.

## Running the Video Analyzer App (Recommended)

The new **Electron desktop app** provides a beautiful Premiere Pro-inspired interface for video analysis and semantic search:

```bash
cd video-analyzer-app
npm install
npm run electron
```

This will:
- Start the Vite dev server (port 5173)
- Start the Python Flask API (port 5001)
- Open the Electron app window

**Features:**
- 🔍 Semantic search by vibe, objects, mood, location
- 🎨 Modern dark theme with smooth animations
- 📊 Thumbnail grid with frame-level metadata
- ⚡ Fast TF-IDF-based search matching

See `ELECTRON_APP_SUMMARY.md` for detailed documentation.

## Running the Core Pipeline (Legacy)

Prepare your assets:

1. Place source videos in `data/clips/` (create the folder if it does not exist).
2. Provide a narration script at `data/script/narration_script.txt` (or update the paths you pass to the scripts).

Then run the orchestrator:

```
python src/main.py
```

`main.py` will:

1. Generate narration audio via ElevenLabs (`src/scriptToTTS.py`).
2. Match script sentences to relevant clips based on metadata (`src/scriptMatching.py`).
3. Assemble the final edit with synchronized audio (`src/videoProcessing.py`).

The rendered video is written to `outputs/final_video.mp4`.

### Running Individual Steps

Every stage is a standalone script. For example, to regenerate the narration only:

```
python src/scriptToTTS.py
```

To rerun the editor with custom paths:

```
python src/videoProcessing.py data/matches/matched_clips.json data/audio/narration_audio.mp3 data/clips outputs/final_video.mp4
```

## Voiceover Web Prototype

```
cd voiceover_app
npm install
npm run build # or npm start for development

# Start the backing Flask server
python src/server.py
```

The Flask server expects Amazon Polly credentials (see environment variables above) and serves the React build output.

## Tests and Linting

Automated tests are not yet configured. Recommended manual checks:

- Run `python src/main.py` end-to-end with a small clip set.
- Manually review generated narration and clip alignment.
- For the web app, run the Flask server locally and verify that the Polly voice list loads correctly.

## Creating a Fresh GitHub Repository

1. Create a new empty repository on GitHub (without any starter files).
2. In this project directory, update the remote and push:
   ```
   git remote remove origin
   git remote add origin git@github.com:<your-username>/<new-repo>.git
   git branch -M main
   git push -u origin main
   ```

Rotate any credentials that were committed prior to sanitizing the repository.
