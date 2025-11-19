"""Simple Tkinter UI for the AI color grading prototype.

The UI keeps all grading work local by delegating to the `AIColorGrader`
class, which applies OpenCV/NumPy adjustments after receiving guidance from an
LLM. The assistant only suggests adjustment parameters—the image processing
itself happens locally.
"""

from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Sequence


# Ensure we can import the sibling module even when launched directly.
APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from ai_color_grader import AIColorGrader, IterationResult  # noqa: E402


class ColorGraderUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AI Color Grader")
        self.root.geometry("960x720")

        self.grader = AIColorGrader()

        self.selected_image: Path | None = None
        self.output_directory: Path | None = None
        self.iterations: Sequence[IterationResult] = []
        self.photo_cache: Dict[int, tk.PhotoImage] = {}

        self._build_layout()

        self.progress_queue: queue.Queue[str] = queue.Queue()
        self.root.after(200, self._poll_progress)

    # ------------------------------------------------------------------
    # UI construction

    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=15)
        main.pack(fill=tk.BOTH, expand=True)

        # Image selector
        file_frame = ttk.Frame(main)
        file_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(file_frame, text="Source image:").pack(side=tk.LEFT)
        self.image_path_var = tk.StringVar(value="No file selected")
        ttk.Button(file_frame, text="Choose...", command=self._choose_image).pack(
            side=tk.LEFT, padx=(8, 12)
        )
        ttk.Label(file_frame, textvariable=self.image_path_var, wraplength=600).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

        # Output directory selector
        out_frame = ttk.Frame(main)
        out_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(out_frame, text="Output folder:").pack(side=tk.LEFT)
        self.output_path_var = tk.StringVar(value="Will reuse source folder")
        ttk.Button(out_frame, text="Choose...", command=self._choose_output_dir).pack(
            side=tk.LEFT, padx=(8, 12)
        )
        ttk.Label(out_frame, textvariable=self.output_path_var, wraplength=600).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

        # Creative brief input
        brief_frame = ttk.Frame(main)
        brief_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(brief_frame, text="Creative brief / instructions:").pack(anchor=tk.W)
        self.brief_text = tk.Text(brief_frame, height=4, wrap=tk.WORD)
        self.brief_text.pack(fill=tk.X)

        # Generation controls
        control_frame = ttk.Frame(main)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(control_frame, text="Max iterations:").pack(side=tk.LEFT)
        self.max_iterations_var = tk.StringVar(value="6")
        ttk.Spinbox(
            control_frame,
            from_=1,
            to=15,
            textvariable=self.max_iterations_var,
            width=4,
        ).pack(side=tk.LEFT, padx=(6, 16))

        self.run_button = ttk.Button(control_frame, text="Run Color Grading", command=self._run)
        self.run_button.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(control_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=(12, 0))

        ttk.Separator(main).pack(fill=tk.X, pady=10)

        # Iteration viewer
        viewer_frame = ttk.Frame(main)
        viewer_frame.pack(fill=tk.BOTH, expand=True)

        list_frame = ttk.Frame(viewer_frame)
        list_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(list_frame, text="Iterations:").pack(anchor=tk.W)
        self.iteration_list = tk.Listbox(list_frame, height=10)
        self.iteration_list.pack(fill=tk.Y, expand=True)
        self.iteration_list.bind("<<ListboxSelect>>", self._on_iteration_selected)

        display_frame = ttk.Frame(viewer_frame)
        display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0))

        self.image_label = ttk.Label(display_frame)
        self.image_label.pack(pady=(0, 12))

        self.details_var = tk.StringVar(value="Run grading to view results.")
        ttk.Label(
            display_frame,
            textvariable=self.details_var,
            justify=tk.LEFT,
            wraplength=520,
        ).pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Event handlers

    def _choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.selected_image = Path(path)
            self.image_path_var.set(str(self.selected_image))
            if not self.output_directory:
                self.output_path_var.set("Will reuse source folder")

    def _choose_output_dir(self) -> None:
        directory = filedialog.askdirectory(title="Choose output folder")
        if directory:
            self.output_directory = Path(directory)
            self.output_path_var.set(str(self.output_directory))

    def _run(self) -> None:
        if not self.selected_image:
            messagebox.showerror("Missing image", "Please choose an image to grade.")
            return

        creative_brief = self.brief_text.get("1.0", tk.END).strip()
        if not creative_brief:
            messagebox.showerror("Missing brief", "Please describe the look you want.")
            return

        try:
            max_iterations = max(1, int(self.max_iterations_var.get()))
        except ValueError:
            messagebox.showerror("Invalid iterations", "Max iterations must be a whole number.")
            return

        output_dir = self.output_directory or self.selected_image.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        suffix = self.selected_image.suffix if self.selected_image.suffix else ".jpg"
        output_path = output_dir / f"{self.selected_image.stem}_graded{suffix}"
        counter = 1
        while output_path.exists():
            output_path = output_dir / f"{self.selected_image.stem}_graded_{counter}{suffix}"
            counter += 1

        self._set_running(True)
        self.status_var.set("Running...")

        thread = threading.Thread(
            target=self._run_grading_thread,
            args=(
                self.selected_image,
                creative_brief,
                output_path,
                max_iterations,
            ),
            daemon=True,
        )
        thread.start()

    # ------------------------------------------------------------------
    # Background processing

    def _run_grading_thread(
        self,
        image_path: Path,
        creative_brief: str,
        output_path: Path,
        max_iterations: int,
    ) -> None:
        try:
            history = self.grader.run(
                input_path=image_path,
                creative_brief=creative_brief,
                output_path=output_path,
                max_iterations=max_iterations,
            )
        except Exception as exc:  # pragma: no cover - runtime specific
            self.progress_queue.put(f"ERROR::{exc}")
            return

        self.progress_queue.put(("DONE", history))

    def _poll_progress(self) -> None:
        try:
            while True:
                item = self.progress_queue.get_nowait()
                if isinstance(item, str) and item.startswith("ERROR::"):
                    self._handle_error(item.split("ERROR::", 1)[1])
                elif isinstance(item, tuple) and item[0] == "DONE":
                    history = item[1]
                    self._handle_done(history)
        except queue.Empty:
            pass
        finally:
            self.root.after(200, self._poll_progress)

    def _handle_error(self, error_message: str) -> None:
        self._set_running(False)
        self.status_var.set("Idle")
        messagebox.showerror("Color grader error", error_message)

    def _handle_done(self, history: Sequence[IterationResult]) -> None:
        self.iterations = history
        self.photo_cache.clear()
        self.iteration_list.delete(0, tk.END)

        for entry in history:
            label = f"Iteration {entry.iteration}"
            if entry.review_status and entry.review_status not in {"complete", "approved"}:
                label += f" ({entry.review_status})"
            self.iteration_list.insert(tk.END, label)

        if history:
            self.iteration_list.select_set(0)
            self._display_iteration(history[0])
            self.status_var.set("Done")
        else:
            self.details_var.set("No iterations returned. Check logs for details.")
            self.status_var.set("Done")

        self._set_running(False)

    # ------------------------------------------------------------------
    # Iteration display helpers

    def _on_iteration_selected(self, _event: tk.Event) -> None:
        if not self.iterations:
            return
        selection = self.iteration_list.curselection()
        if not selection:
            return
        index = selection[0]
        if index < len(self.iterations):
            self._display_iteration(self.iterations[index])

    def _display_iteration(self, entry: IterationResult) -> None:
        from PIL import Image, ImageTk  # Lazy import to avoid hard dependency until needed

        image_path = entry.output_path
        if not image_path or not Path(image_path).exists():
            self.details_var.set("Iteration image is missing on disk.")
            return

        cache_key = entry.iteration
        if cache_key not in self.photo_cache:
            img = Image.open(image_path)
            resample_attr = getattr(Image, "Resampling", Image)
            resample = getattr(resample_attr, "LANCZOS", getattr(Image, "BICUBIC", Image.NEAREST))
            img.thumbnail((720, 480), resample)
            self.photo_cache[cache_key] = ImageTk.PhotoImage(img)

        self.image_label.configure(image=self.photo_cache[cache_key])

        details_lines: List[str] = [
            f"Iteration {entry.iteration}",
            f"Status: {entry.review_status or 'n/a'}",
        ]

        if entry.review_feedback:
            details_lines.append(f"Feedback: {entry.review_feedback}")

        if entry.adjustments:
            details_lines.append("Adjustments:")
            for adj in entry.adjustments:
                value_str = f" {adj.value:+.3f}" if adj.value is not None else ""
                params_str = (
                    " (" + ", ".join(f"{k}={v}" for k, v in adj.params.items()) + ")"
                    if adj.params
                    else ""
                )
                line = f"  • {adj.type}:{value_str}{params_str}"
                if adj.comment:
                    line += f" ({adj.comment})"
                details_lines.append(line)
        else:
            details_lines.append("Adjustments: none recorded")

        details_lines.append(f"File: {image_path}")
        self.details_var.set("\n".join(details_lines))

    # ------------------------------------------------------------------
    # Utility helpers

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "!disabled"
        self.run_button.state((state,))


def main() -> None:
    root = tk.Tk()
    ColorGraderUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()


