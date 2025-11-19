"""Prototype for an AI-assisted color grading workflow on a single image.

This module keeps the prototype decoupled from the core video project so it can
be iterated on independently before integration. It implements a simple loop:

1. Send the original frame plus a creative brief to an LLM. The model responds
   with a JSON plan describing supported color adjustments.
2. Apply the requested adjustments locally using OpenCV / NumPy.
3. Send the graded result back to the LLM for review. The model can accept the
   grade or supply another round of refinements. The loop repeats for a bounded
   number of iterations.

Only basic adjustments are implemented to keep the first version predictable.
Future work can extend both the prompt schema and the adjustment engine.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

import colour
import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables (e.g. OPENAI_API_KEY) once on import.
load_dotenv()


SUPPORTED_ADJUSTMENTS = {
    "exposure",
    "contrast",
    "saturation",
    "vibrance",
    "temperature",
    "tint",
    "gamma",
    "lift",
    "gamma_midtone",
    "gain",
    "shadows",
    "highlights",
    "curve_master",
    "curve_red",
    "curve_green",
    "curve_blue",
}

ADJUSTMENT_RANGES: Dict[str, tuple[float, float]] = {
    "exposure": (-2.0, 2.0),
    "contrast": (-1.0, 1.0),
    "saturation": (-1.0, 1.0),
    "vibrance": (-1.0, 1.0),
    "temperature": (-1.0, 1.0),
    "tint": (-1.0, 1.0),
    "gamma": (-0.9, 1.0),
    "lift": (-0.5, 0.5),
    "gamma_midtone": (-0.5, 0.5),
    "gain": (-0.5, 0.5),
    "shadows": (-0.5, 0.5),
    "highlights": (-0.5, 0.5),
}


@dataclass
class Adjustment:
    """Represents a single color adjustment step."""

    type: str
    value: float | None = None
    comment: str | None = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IterationResult:
    iteration: int
    adjustments: List[Adjustment] = field(default_factory=list)
    review_status: str | None = None
    review_feedback: str | None = None
    output_path: Path | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "output_path": str(self.output_path) if self.output_path else None,
            "review_status": self.review_status,
            "review_feedback": self.review_feedback,
            "adjustments": [
                {
                    "type": adj.type,
                    "value": adj.value,
                    "comment": adj.comment,
                    "params": adj.params,
                }
                for adj in self.adjustments
            ],
        }


class ColorAdjustmentEngine:
    """Applies a limited set of color grading adjustments to an RGB image."""

    def apply(self, image_rgb: np.ndarray, adjustments: Sequence[Adjustment]) -> np.ndarray:
        result = image_rgb.astype(np.float32) / 255.0
        for adj in adjustments:
            handler_name = f"_apply_{adj.type}"
            if not hasattr(self, handler_name):
                raise ValueError(f"Adjustment '{adj.type}' is not supported by the engine.")

            handler = getattr(self, handler_name)
            result = handler(result, adj)

        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        return result

    @staticmethod
    def _apply_exposure(image: np.ndarray, adjustment: Adjustment) -> np.ndarray:
        value = float(adjustment.value or 0.0)
        factor = 1.0 + value
        return np.clip(image * factor, 0.0, 1.0)

    @staticmethod
    def _apply_contrast(image: np.ndarray, adjustment: Adjustment) -> np.ndarray:
        value = float(adjustment.value or 0.0)
        factor = 1.0 + value
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        return np.clip((image - mean) * factor + mean, 0.0, 1.0)

    @staticmethod
    def _apply_saturation(image: np.ndarray, adjustment: Adjustment) -> np.ndarray:
        value = float(adjustment.value or 0.0)
        hsv = cv2.cvtColor((image * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + value), 0.0, 255.0)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return result.astype(np.float32) / 255.0

    @staticmethod
    def _apply_vibrance(image: np.ndarray, adjustment: Adjustment) -> np.ndarray:
        value = float(adjustment.value or 0.0)
        if abs(value) < 1e-6:
            return image

        hsv = colour.models.RGB_to_HSV(image)
        saturation = hsv[..., 1]
        delta = value * (1.0 - saturation)
        hsv[..., 1] = np.clip(saturation + delta, 0.0, 1.0)
        result = colour.models.HSV_to_RGB(hsv)
        return np.clip(result, 0.0, 1.0)

    @staticmethod
    def _apply_temperature(image: np.ndarray, adjustment: Adjustment) -> np.ndarray:
        value = float(adjustment.value or 0.0)
        warm_shift = value * 0.15
        result = image.copy()
        result[..., 0] = np.clip(result[..., 0] + warm_shift, 0.0, 1.0)  # Red channel
        result[..., 2] = np.clip(result[..., 2] - warm_shift, 0.0, 1.0)  # Blue channel
        return result

    @staticmethod
    def _apply_tint(image: np.ndarray, adjustment: Adjustment) -> np.ndarray:
        value = float(adjustment.value or 0.0)
        tint_shift = value * 0.1
        result = image.copy()
        result[..., 0] = np.clip(result[..., 0] + tint_shift, 0.0, 1.0)
        result[..., 1] = np.clip(result[..., 1] - tint_shift, 0.0, 1.0)
        result[..., 2] = np.clip(result[..., 2] + tint_shift, 0.0, 1.0)
        return result

    @staticmethod
    def _apply_gamma(image: np.ndarray, adjustment: Adjustment) -> np.ndarray:
        value = float(adjustment.value or 0.0)
        gamma = max(0.01, 1.0 + value)
        inv_gamma = 1.0 / gamma
        return np.power(np.clip(image, 0.0, 1.0), inv_gamma)

    @staticmethod
    def _apply_lift(image: np.ndarray, adjustment: Adjustment) -> np.ndarray:
        value = float(adjustment.value or 0.0)
        if abs(value) < 1e-6:
            return image
        luma = np.dot(image, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
        weight = np.power(1.0 - luma, 1.5)[..., None]
        return np.clip(image + value * weight, 0.0, 1.0)

    @staticmethod
    def _apply_gamma_midtone(image: np.ndarray, adjustment: Adjustment) -> np.ndarray:
        value = float(adjustment.value or 0.0)
        if abs(value) < 1e-6:
            return image
        gamma = max(0.01, 1.0 + value)
        luma = np.dot(image, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
        weight = np.clip(1.0 - np.abs(luma - 0.5) * 2.0, 0.0, 1.0)[..., None]
        adjusted = np.power(np.clip(image, 0.0, 1.0), 1.0 / gamma)
        return np.clip(image * (1.0 - weight) + adjusted * weight, 0.0, 1.0)

    @staticmethod
    def _apply_gain(image: np.ndarray, adjustment: Adjustment) -> np.ndarray:
        value = float(adjustment.value or 0.0)
        if abs(value) < 1e-6:
            return image
        luma = np.dot(image, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
        weight = np.power(luma, 1.5)[..., None]
        return np.clip(image + value * weight, 0.0, 1.0)

    @staticmethod
    def _apply_shadows(image: np.ndarray, adjustment: Adjustment) -> np.ndarray:
        value = float(adjustment.value or 0.0)
        if abs(value) < 1e-6:
            return image
        luma = np.dot(image, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
        weight = np.power(1.0 - luma, 2.0)[..., None]
        return np.clip(image + value * weight, 0.0, 1.0)

    @staticmethod
    def _apply_highlights(image: np.ndarray, adjustment: Adjustment) -> np.ndarray:
        value = float(adjustment.value or 0.0)
        if abs(value) < 1e-6:
            return image
        luma = np.dot(image, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
        weight = np.power(luma, 2.0)[..., None]
        return np.clip(image + value * weight, 0.0, 1.0)

    def _apply_curve_master(self, image: np.ndarray, adjustment: Adjustment) -> np.ndarray:
        points = self._normalise_curve_points(adjustment)
        if points is None:
            return image
        curve = self._generate_curve_lut(points)
        mapped = self._apply_curve_lut(image, curve)
        return np.clip(mapped, 0.0, 1.0)

    def _apply_curve_red(self, image: np.ndarray, adjustment: Adjustment) -> np.ndarray:
        points = self._normalise_curve_points(adjustment)
        if points is None:
            return image
        curve = self._generate_curve_lut(points)
        mapped = image.copy()
        mapped[..., 0] = self._apply_curve_channel(mapped[..., 0], curve)
        return np.clip(mapped, 0.0, 1.0)

    def _apply_curve_green(self, image: np.ndarray, adjustment: Adjustment) -> np.ndarray:
        points = self._normalise_curve_points(adjustment)
        if points is None:
            return image
        curve = self._generate_curve_lut(points)
        mapped = image.copy()
        mapped[..., 1] = self._apply_curve_channel(mapped[..., 1], curve)
        return np.clip(mapped, 0.0, 1.0)

    def _apply_curve_blue(self, image: np.ndarray, adjustment: Adjustment) -> np.ndarray:
        points = self._normalise_curve_points(adjustment)
        if points is None:
            return image
        curve = self._generate_curve_lut(points)
        mapped = image.copy()
        mapped[..., 2] = self._apply_curve_channel(mapped[..., 2], curve)
        return np.clip(mapped, 0.0, 1.0)

    @staticmethod
    def _normalise_curve_points(adjustment: Adjustment) -> np.ndarray | None:
        points = adjustment.params.get("points") if adjustment.params else None
        if not points:
            return None
        try:
            pts = np.array(points, dtype=np.float32)
        except ValueError:
            return None
        if pts.ndim != 2 or pts.shape[1] != 2:
            return None
        pts = np.clip(pts, 0.0, 1.0)
        pts = pts[np.argsort(pts[:, 0])]
        if pts[0, 0] > 0.0:
            pts = np.vstack(([0.0, pts[0, 1]], pts))
        if pts[-1, 0] < 1.0:
            pts = np.vstack((pts, [1.0, pts[-1, 1]]))
        return pts

    @staticmethod
    def _generate_curve_lut(points: np.ndarray, resolution: int = 1024) -> np.ndarray:
        x = np.linspace(0.0, 1.0, resolution)
        y = np.interp(x, points[:, 0], points[:, 1])
        return np.clip(y, 0.0, 1.0)

    @staticmethod
    def _apply_curve_lut(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
        return np.stack(
            [ColorAdjustmentEngine._apply_curve_channel(image[..., idx], lut) for idx in range(3)],
            axis=-1,
        )

    @staticmethod
    def _apply_curve_channel(channel: np.ndarray, lut: np.ndarray) -> np.ndarray:
        positions = np.clip(channel, 0.0, 1.0) * (len(lut) - 1)
        lower = np.floor(positions).astype(np.int32)
        upper = np.clip(lower + 1, 0, len(lut) - 1)
        blend = positions - lower
        return lut[lower] * (1.0 - blend) + lut[upper] * blend


class AIColorGrader:
    """Coordinates model calls and local adjustments for iterative grading."""

    def __init__(self, plan_model: str = "gpt-4o-mini", review_model: str = "gpt-4o-mini") -> None:
        self.client = OpenAI()
        self.plan_model = plan_model
        self.review_model = review_model
        self.engine = ColorAdjustmentEngine()
        self._structured_outputs_supported = True

    def run(
        self,
        input_path: Path,
        creative_brief: str,
        output_path: Path,
        max_iterations: int = 3,
    ) -> List[IterationResult]:
        input_rgb = self._load_image(input_path)
        original_b64 = self._encode_image(input_rgb)

        history: List[IterationResult] = []
        current_rgb = input_rgb

        plan_payload = self._request_plan(original_b64, creative_brief)
        plan_adjustments = self._parse_adjustments(plan_payload)

        current_rgb = self.engine.apply(current_rgb, plan_adjustments)
        iteration_result = self._store_iteration(
            iteration=1,
            adjustments=plan_adjustments,
            image_rgb=current_rgb,
            output_path=output_path,
        )
        history.append(iteration_result)

        if max_iterations <= 1:
            return history

        for iteration in range(2, max_iterations + 1):
            review = self._request_review(
                original_b64=original_b64,
                graded_b64=self._encode_image(current_rgb),
                creative_brief=creative_brief,
            )

            status = review.get("status", "complete").lower()
            feedback = review.get("feedback") or review.get("comment")

            history[-1].review_status = status
            history[-1].review_feedback = feedback

            if status in {"complete", "approved", "accept"}:
                break

            new_adjustments = self._parse_adjustments(review)
            if not new_adjustments:
                break

            current_rgb = self.engine.apply(current_rgb, new_adjustments)
            iteration_result = self._store_iteration(
                iteration=iteration,
                adjustments=new_adjustments,
                image_rgb=current_rgb,
                output_path=output_path,
            )
            iteration_result.review_feedback = feedback
            history.append(iteration_result)

        return history

    # ---------------------------------------------------------------------
    # OpenAI helpers

    def _request_plan(self, original_b64: str, creative_brief: str) -> Dict[str, Any]:
        system_prompt = (
            "You are a senior colorist. Respond with a compact JSON object describing \n"
            "how to grade the provided image so it matches the creative brief. Use the \n"
            "schema: {\"adjustments\": [{\"type\": <string>, \"value\": <float?>, \"comment\": <string?>, \"points\": <array?>}]}.\n"
            "Valid adjustment types: exposure, contrast, saturation, vibrance, temperature, tint, gamma, \n"
            "lift, gamma_midtone, gain, shadows, highlights, curve_master, curve_red, curve_green, curve_blue.\n"
            "Curves must include 2-8 points as either [x, y] pairs or objects with x/y fields. Do NOT request LUTs."
        )

        user_prompt = (
            "Creative brief: "
            f"{creative_brief}\n"
            "Numeric adjustments should stay within sensible ranges (see schema) and curves must span 0..1. \n"
            "Aim for no more than six steps per pass."
        )

        response = self.client.chat.completions.create(
            model=self.plan_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{original_b64}"}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
        )

        return json.loads(response.choices[0].message.content)

    def _request_review(
        self,
        original_b64: str,
        graded_b64: str,
        creative_brief: str,
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are a strict, professional colorist reviewing a color grade. Reply with JSON using the schema: \n"
            "{\"status\": \"complete|revise\", \"feedback\": <string?>, \"adjustments\": [...] }.\n"
            "Only mark status as 'complete' if the graded image PRECISELY matches the creative brief. \n"
            "If there are ANY color casts (too red, too blue, too green, etc.), incorrect exposure, wrong mood, \n"
            "or any deviation from the brief, mark status as 'revise' and provide corrective adjustments. \n"
            "Be highly critical. Use the adjustment vocabulary: exposure, contrast, saturation, vibrance, temperature, \n"
            "tint, gamma, lift, gamma_midtone, gain, shadows, highlights, curve_master, curve_red, curve_green, curve_blue. \n"
            "No LUTs."
        )

        user_prompt = (
            "Compare the graded frame to the brief. Look carefully for unwanted color casts, incorrect tones, \n"
            "wrong exposure levels, or any mismatch with the requested look. If ANYTHING is off, return 'revise' \n"
            "with specific corrective adjustments. Only return 'complete' if it's perfect."
        )

        response = self.client.chat.completions.create(
            model=self.review_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Creative brief: {creative_brief}\n\n{user_prompt}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{original_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{graded_b64}"}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
        )

        return json.loads(response.choices[0].message.content)

    # ------------------------------------------------------------------
    # Utility helpers

    def _store_iteration(
        self,
        iteration: int,
        adjustments: Sequence[Adjustment],
        image_rgb: np.ndarray,
        output_path: Path,
    ) -> IterationResult:
        output_path = output_path.expanduser().resolve()
        if output_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            raise ValueError("Output path must end with .png, .jpg, or .jpeg")

        if iteration == 1:
            target_path = output_path
        else:
            target_path = output_path.with_name(
                f"{output_path.stem}_iter{iteration}{output_path.suffix}"
            )

        self._save_image(target_path, image_rgb)

        return IterationResult(
            iteration=iteration,
            adjustments=list(adjustments),
            output_path=target_path,
        )

    @staticmethod
    def _parse_adjustments(payload: Dict[str, Any]) -> List[Adjustment]:
        adjustments_raw: Any

        if isinstance(payload, dict) and "adjustments" in payload:
            adjustments_raw = payload.get("adjustments", [])
        elif isinstance(payload, dict):
            adjustments_raw = payload.get("steps", [])
        else:
            adjustments_raw = []

        adjustments: List[Adjustment] = []

        for step in adjustments_raw or []:
            if not isinstance(step, dict):
                continue

            adj_type = step.get("type")
            if not isinstance(adj_type, str):
                continue

            adj_type = adj_type.lower().strip()
            if adj_type not in SUPPORTED_ADJUSTMENTS:
                continue

            comment = step.get("comment") or step.get("reason")
            params: Dict[str, Any] = {}

            if adj_type.startswith("curve_"):
                points_raw = step.get("points") or step.get("curve")
                parsed_points: List[List[float]] = []
                if isinstance(points_raw, list):
                    for item in points_raw:
                        if isinstance(item, dict):
                            x = item.get("x")
                            if x is None:
                                x = item.get("input")
                            y = item.get("y")
                            if y is None:
                                y = item.get("output")
                        elif isinstance(item, (list, tuple)) and len(item) >= 2:
                            x, y = item[0], item[1]
                        else:
                            continue
                        try:
                            parsed_points.append([float(x), float(y)])
                        except (TypeError, ValueError):
                            continue
                if not parsed_points:
                    continue
                params["points"] = parsed_points
                value = None
            else:
                raw_value = step.get("value", 0.0)
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                low, high = ADJUSTMENT_RANGES.get(adj_type, (-1.0, 1.0))
                value = float(np.clip(value, low, high))

            adjustments.append(
                Adjustment(
                    type=adj_type,
                    value=value,
                    comment=comment,
                    params=params,
                )
            )

        return adjustments

    @staticmethod
    def _load_image(image_path: Path) -> np.ndarray:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image at {image_path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _save_image(path: Path, image_rgb: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), image_bgr)

    @staticmethod
    def _encode_image(image_rgb: np.ndarray) -> str:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success:
            raise RuntimeError("Failed to encode image to JPEG")
        return base64.b64encode(buffer).decode("utf-8")

    # ------------------------------------------------------------------
    # OpenAI client helpers

    def _responses_create(self, **kwargs: Any):
        if not self._structured_outputs_supported:
            kwargs.pop("response_format", None)
            return self.client.responses.create(**kwargs)

        try:
            return self.client.responses.create(**kwargs)
        except TypeError as exc:
            if "response_format" in kwargs:
                self._structured_outputs_supported = False
                kwargs.pop("response_format", None)
                return self.client.responses.create(**kwargs)
            raise exc

    @staticmethod
    def _extract_text(response: Any) -> str:
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text

        if hasattr(response, "output"):
            segments: List[str] = []
            for item in getattr(response, "output") or []:
                content = getattr(item, "content", None)
                if not content:
                    continue
                for block in content:
                    text = block.get("text") if isinstance(block, dict) else None
                    if text:
                        segments.append(text)
            if segments:
                return "\n".join(segments)

        if hasattr(response, "choices"):
            segments = []
            for choice in getattr(response, "choices") or []:
                message = getattr(choice, "message", None)
                if message and isinstance(message, dict):
                    text = message.get("content")
                    if isinstance(text, str):
                        segments.append(text)
            if segments:
                return "\n".join(segments)

        raise ValueError("Unable to extract text content from OpenAI response")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI-guided color grading prototype")
    parser.add_argument("--image", required=True, type=Path, help="Path to the input image")
    parser.add_argument(
        "--brief",
        required=True,
        help="Creative direction for the grade (e.g. 'moody teal and orange at dusk')",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Where to write the graded frame (iteration suffixes are added for revisions)",
    )
    parser.add_argument("--max-iterations", type=int, default=3, help="Max review cycles to run")
    parser.add_argument(
        "--plan-model",
        default="gpt-4o-mini",
        help="Model used to draft the initial grading plan",
    )
    parser.add_argument(
        "--review-model",
        default="gpt-4o-mini",
        help="Model used for iterative review feedback",
    )
    parser.add_argument(
        "--session-log",
        type=Path,
        help="Optional path to write a JSON log summarizing each iteration",
    )
    parser.add_argument(
        "--html-report",
        type=Path,
        help="Optional path to write a static HTML viewer for the session",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch a simple Tkinter UI to browse grading iterations",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    grader = AIColorGrader(plan_model=args.plan_model, review_model=args.review_model)
    history = grader.run(
        input_path=args.image,
        creative_brief=args.brief,
        output_path=args.output,
        max_iterations=args.max_iterations,
    )

    if args.session_log:
        _write_session_log(args.session_log, history)

    if args.html_report:
        _generate_html_report(args.html_report, history, creative_brief=args.brief)

    if args.ui:
        _launch_simple_ui(history, creative_brief=args.brief)

    print("\nColor grading session summary:")
    for iteration in history:
        print(f"- Iteration {iteration.iteration} -> {iteration.output_path}")
        for adj in iteration.adjustments:
            comment = f" ({adj.comment})" if adj.comment else ""
            value_str = f"{adj.value:+.3f}" if adj.value is not None else ""
            params_str = (
                " "
                + ", ".join(f"{k}={v}" for k, v in adj.params.items())
                if adj.params
                else ""
            )
            print(f"    * {adj.type}:{value_str}{params_str}{comment}")
        if iteration.review_status:
            print(f"    review_status: {iteration.review_status}")
        if iteration.review_feedback:
            print(f"    feedback: {iteration.review_feedback}")

    return 0


def _write_session_log(path: Path, history: Sequence[IterationResult]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"iterations": [entry.to_dict() for entry in history]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _generate_html_report(
    path: Path,
    history: Sequence[IterationResult],
    creative_brief: str,
) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[str] = []
    for entry in history:
        if not entry.output_path:
            continue

        rel_path = os.path.relpath(entry.output_path, path.parent)
        adjustments_html_parts: List[str] = []
        for adj in entry.adjustments:
            value_str = f" {adj.value:+.2f}" if adj.value is not None else ""
            params_str = (
                " ("
                + ", ".join(f"{k}={v}" for k, v in adj.params.items())
                + ")"
                if adj.params
                else ""
            )
            comment = f" – {adj.comment}" if adj.comment else ""
            adjustments_html_parts.append(
                f"<li><strong>{adj.type}</strong>{value_str}{params_str}{comment}</li>"
            )
        adjustments_html = "".join(adjustments_html_parts) or "<li>No adjustments recorded</li>"

        feedback_html = (
            f"<p class='feedback'><strong>Feedback:</strong> {entry.review_feedback}</p>"
            if entry.review_feedback
            else ""
        )

        rows.append(
            f"""
            <section class="iteration">
              <h2>Iteration {entry.iteration}</h2>
              <img src="{rel_path}" alt="Iteration {entry.iteration}"
                   loading="lazy" />
              <div class="details">
                <p><strong>Status:</strong> {entry.review_status or 'n/a'}</p>
                {feedback_html}
                <p><strong>Adjustments</strong></p>
                <ul>{adjustments_html}</ul>
              </div>
            </section>
            """
        )

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <title>AI Color Grading Session</title>
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; background: #f5f5f5; }}
        header {{ margin-bottom: 2rem; }}
        header h1 {{ margin: 0 0 0.5rem 0; }}
        .iteration {{ background: white; padding: 1.5rem; margin-bottom: 2rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
        .iteration img {{ max-width: 100%; border-radius: 8px; margin: 1rem 0; }}
        .details {{ font-size: 0.95rem; color: #333; }}
        .details ul {{ padding-left: 1.25rem; }}
        .feedback {{ margin: 0.5rem 0; color: #555; }}
      </style>
    </head>
    <body>
      <header>
        <h1>AI Color Grading Session</h1>
        <p><strong>Creative brief:</strong> {creative_brief}</p>
      </header>
      {''.join(rows)}
    </body>
    </html>
    """

    path.write_text(html, encoding="utf-8")


def _launch_simple_ui(history: Sequence[IterationResult], creative_brief: str) -> None:
    iterations = [entry for entry in history if entry.output_path]
    if not iterations:
        print("No iterations with output frames to display in the UI.")
        return

    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:  # pragma: no cover - depends on environment
        print(f"Tkinter is required for --ui but is not available: {exc}")
        return

    try:
        from PIL import Image, ImageTk
    except ImportError as exc:  # pragma: no cover - depends on environment
        print(f"Pillow is required for --ui but is not available: {exc}")
        return

    root = tk.Tk()
    root.title("AI Color Grader – Iteration Viewer")

    header = ttk.Label(
        root,
        text=f"Creative brief: {creative_brief}",
        padding=(10, 10, 10, 5),
        wraplength=720,
        justify=tk.LEFT,
    )
    header.pack(fill=tk.X)

    image_label = ttk.Label(root)
    image_label.pack(padx=10, pady=10)

    info_var = tk.StringVar()
    info_label = ttk.Label(
        root,
        textvariable=info_var,
        justify=tk.LEFT,
        padding=(10, 0, 10, 10),
        wraplength=720,
    )
    info_label.pack(fill=tk.X)

    nav_frame = ttk.Frame(root)
    nav_frame.pack(pady=(0, 10))

    prev_button = ttk.Button(nav_frame, text="◀ Previous")
    prev_button.grid(row=0, column=0, padx=5)
    next_button = ttk.Button(nav_frame, text="Next ▶")
    next_button.grid(row=0, column=1, padx=5)

    state: Dict[str, Any] = {"index": 0, "photo": None}

    def load_image(image_path: Path) -> ImageTk.PhotoImage:
        resample_attr = getattr(Image, "Resampling", Image)
        resample = getattr(resample_attr, "LANCZOS", getattr(Image, "BICUBIC", Image.NEAREST))
        img = Image.open(image_path)
        img.thumbnail((960, 540), resample)
        return ImageTk.PhotoImage(img)

    def format_details(entry: IterationResult) -> str:
        lines = [
            f"Iteration {entry.iteration}",
            f"Status: {entry.review_status or 'n/a'}",
        ]
        if entry.review_feedback:
            lines.append(f"Feedback: {entry.review_feedback}")
        if entry.adjustments:
            lines.append("Adjustments:")
            for adj in entry.adjustments:
                line = f"  • {adj.type}: {adj.value:+.2f}"
                if adj.comment:
                    line += f" ({adj.comment})"
                lines.append(line)
        else:
            lines.append("Adjustments: none recorded")
        lines.append(f"File: {entry.output_path}")
        return "\n".join(lines)

    def update_view(index: int) -> None:
        entry = iterations[index]
        try:
            photo = load_image(entry.output_path)
        except Exception as exc:  # pragma: no cover - depends on files on disk
            info_var.set(f"Failed to load image: {exc}")
            return

        image_label.configure(image=photo)
        state["photo"] = photo  # Prevent garbage collection
        info_var.set(format_details(entry))
        state["index"] = index
        prev_button.state(("!disabled",) if index > 0 else ("disabled",))
        next_button.state(("!disabled",) if index < len(iterations) - 1 else ("disabled",))

    def go_prev() -> None:
        if state["index"] > 0:
            update_view(state["index"] - 1)

    def go_next() -> None:
        if state["index"] < len(iterations) - 1:
            update_view(state["index"] + 1)

    prev_button.configure(command=go_prev)
    next_button.configure(command=go_next)

    try:
        update_view(0)
    except Exception as exc:  # pragma: no cover - depends on files on disk
        print(f"Unable to display the UI: {exc}")
        root.destroy()
        return

    root.mainloop()


if __name__ == "__main__":
    raise SystemExit(main())


