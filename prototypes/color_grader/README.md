# AI Color Grader Prototype

This folder contains an experimental, stand-alone workflow for iteratively color
grading a single image with help from GPT. The goal is to validate the flow and
prompt design before wiring it into the main video pipeline.

## Quick Start

1. Install dependencies (the root `requirements.txt` now includes
   `colour-science` alongside OpenCV/NumPy) and export `OPENAI_API_KEY` in your
   environment. The script reads from `.env` via `python-dotenv` if present.
2. Run the CLI against a still frame, or launch the simple desktop UI.

### CLI Flow

Run the standard loop from the command line:

   ```bash
   python prototypes/color_grader/ai_color_grader.py \
     --image /absolute/path/to/frame.jpg \
     --brief "Moody dusk look with soft highlights" \
     --output /absolute/path/to/graded_frame.jpg \
     --max-iterations 3 \
     --session-log /absolute/path/to/session.json \
     --html-report /absolute/path/to/session.html \
     --ui
   ```

3. The script saves the first grade to the requested output path. Each review
   iteration writes a suffixed copy (e.g. `_iter2`). The console log summarizes
   the adjustments that were applied and any reviewer feedback from the model.
   If you pass `--session-log` or `--html-report`, the run also produces a JSON
   log and/or a static HTML viewer that displays each iteration side by side.
   With `--ui`, a minimal Tkinter window opens so you can browse each iteration
   without leaving Python.

### GUI Flow

Run a very simple Tkinter application that lets you choose the source image,
describe the desired look, and trigger grading with a button:

```bash
python prototypes/color_grader/ui_app.py
```

Steps inside the app:

- Click **Choose...** to pick a still frame (this acts like uploading the
  image). Optionally choose an output folder.
- Enter the creative direction in the text box and adjust the iteration limit
  if needed.
- Press **Run Color Grading**; the app asks GPT for adjustment guidance but all
  image edits run locally via NumPy + `colour-science` operators (no LUTs yet).
- When the run finishes, every iteration appears in the list. Select any entry
  to preview the graded frame and see the adjustments/feedback that produced it.

## How It Works

- **Planning step** – Sends the original image plus your creative brief to the
  model. The system prompt constrains the model to return a JSON plan drawn from
  a DaVinci-style vocabulary: exposure/contrast, saturation/vibrance,
  temperature/tint, gamma, lift/gamma/gain, shadows/highlights, and master/RGB
  curves (2–8 control points each). No LUTs are requested yet.
- **Local grading** – Applies adjustments via NumPy and `colour-science`, keeping
  everything in floating-point sRGB before writing 8-bit outputs. This keeps the
  work reproducible and lets us evolve the math without touching prompts.
- **Review loop** – The graded frame and original are shown to the model. It can
  accept the grade (`status: complete`) or request another pass with additional
  adjustments. The loop stops when the model approves or the iteration cap is
  reached.

## Next Steps

- Explore HDR pipelines (16-bit float) and optional LUT support via
  OpenColorIO.
- Add batch runners for directories of frames and, eventually, whole video
  clips.
- Integrate playback/video previews in the UI and expose an API suitable for
  the broader project.


