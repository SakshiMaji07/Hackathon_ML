from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np


class FPSCounter:
    """Track a rolling, smoothed frames-per-second estimate."""

    def __init__(self, window_size: int = 30):
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")
        self.window_size = int(window_size)
        self.frame_deltas: deque[float] = deque(maxlen=self.window_size)
        self._last_timestamp: float | None = None

    def update(self) -> None:
        """Record the current frame time and its delta from the previous update."""
        now = time.time()
        if self._last_timestamp is not None:
            delta = now - self._last_timestamp
            if delta > 0:
                self.frame_deltas.append(delta)
        self._last_timestamp = now

    def get_fps(self) -> float:
        """Return the reciprocal of the mean interval between recorded frames."""
        if not self.frame_deltas:
            return 0.0
        total_delta = sum(self.frame_deltas)
        if total_delta <= 0:
            return 0.0
        return len(self.frame_deltas) / total_delta


class OverlayRenderer:
    """Draw a compact diagnostics HUD on a BGR image."""

    def __init__(
        self,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.6,
        color: tuple[int, int, int] = (255, 255, 255),
        thickness: int = 1,
        position: tuple[int, int] = (12, 28),
        line_spacing: int = 8,
        background_color: tuple[int, int, int] = (0, 0, 0),
        background_alpha: float = 0.55,
    ):
        if font_scale <= 0:
            raise ValueError("font_scale must be positive.")
        if thickness < 1:
            raise ValueError("thickness must be at least 1.")
        if line_spacing < 0:
            raise ValueError("line_spacing cannot be negative.")
        if not 0.0 <= background_alpha <= 1.0:
            raise ValueError("background_alpha must be between 0 and 1.")
        self.font = font
        self.font_scale = font_scale
        self.color = color
        self.thickness = thickness
        self.position = position
        self.line_spacing = line_spacing
        self.background_color = background_color
        self.background_alpha = background_alpha

    def draw_hud(
        self,
        canvas: np.ndarray,
        fps: float,
        z_order: str,
        status_text: str,
    ) -> np.ndarray:
        """Burn FPS, depth order, and diagnostic text into a BGR canvas."""
        if not isinstance(canvas, np.ndarray) or canvas.ndim != 3 or canvas.shape[2] != 3:
            raise ValueError("canvas must be an HxWx3 BGR NumPy array.")
        if canvas.size == 0:
            raise ValueError("canvas cannot be empty.")

        height, width = canvas.shape[:2]
        margin = max(1, min(self.position[0], self.position[1]))
        lines = [f"FPS: {fps:.1f}", f"Depth: {z_order}", str(status_text)]
        text_height = cv2.getTextSize(
            "Ag",
            self.font,
            self.font_scale,
            self.thickness,
        )[0][1]
        line_height = text_height + self.line_spacing
        x = min(max(0, self.position[0]), max(0, width - 1))
        first_baseline = min(max(text_height, self.position[1]), max(text_height, height - 1))

        # Bound text to the available width rather than letting it clip past the frame.
        available_width = max(1, width - x - margin)
        displayed_lines: list[str] = []
        for line in lines:
            if cv2.getTextSize(line, self.font, self.font_scale, self.thickness)[0][0] <= available_width:
                displayed_lines.append(line)
                continue

            ellipsis = "..."
            truncated = line
            while truncated and cv2.getTextSize(
                truncated + ellipsis,
                self.font,
                self.font_scale,
                self.thickness,
            )[0][0] > available_width:
                truncated = truncated[:-1]
            displayed_lines.append(truncated + ellipsis if truncated else ellipsis)

        baselines = [
            min(first_baseline + index * line_height, height - 1)
            for index in range(len(displayed_lines))
        ]
        max_text_width = max(
            cv2.getTextSize(line, self.font, self.font_scale, self.thickness)[0][0]
            for line in displayed_lines
        )
        box_top = max(0, baselines[0] - text_height - 4)
        box_bottom = min(height - 1, baselines[-1] + 4)
        box_right = min(width - 1, x + max_text_width + 4)

        if self.background_alpha > 0:
            overlay = canvas.copy()
            cv2.rectangle(
                overlay,
                (x, box_top),
                (box_right, box_bottom),
                self.background_color,
                thickness=cv2.FILLED,
            )
            cv2.addWeighted(
                overlay,
                self.background_alpha,
                canvas,
                1.0 - self.background_alpha,
                0,
                dst=canvas,
            )

        for line, baseline in zip(displayed_lines, baselines):
            cv2.putText(
                canvas,
                line,
                (x, baseline),
                self.font,
                self.font_scale,
                self.color,
                self.thickness,
                lineType=cv2.LINE_AA,
            )
        return canvas