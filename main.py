from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

from src.compositing.alpha_blender import CanvasComposer
from src.enhancement.color_harmonizer import ColorHarmonizer
from src.enhancement.depth_estimation import DepthEstimator
from src.input.stream_capture import DualStreamCapture
from src.segmentation.inference import MattingEngine
from src.utils.cuda_utils import CudaStreamManager
from src.utils.visualizer import FPSCounter, OverlayRenderer


LOGGER = logging.getLogger("photobooth")


class PhotoboothApp:
    """Orchestrate dual-camera matting, enhancement, compositing, and display."""

    def __init__(self, config_path):
        self.config_path = Path(config_path).expanduser().resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        if not isinstance(config, dict):
            raise ValueError("The YAML configuration must contain a mapping at its root.")
        self.config: dict[str, Any] = config
        self.project_dir = self.config_path.parent.parent
        self.camera_config = self._section("camera")
        self.model_config = self._section("model")
        self.background_config = self._section("background")
        self.depth_config = self._section("depth")
        self.enhancement_config = self._section("enhancement")
        self.device = self._select_device(
            str(config.get("device", self.model_config.get("device", "cuda")))
        )
        self._configure_cuda_memory()

        self.camera_indices = self._get_camera_indices()
        self.resolution = self._get_resolution()
        self.fps = self._positive_number(
            self.camera_config.get("fps", config.get("fps", 30)),
            "camera.fps",
        )
        self.model_path = self._resolve_existing_file(
            self.model_config.get("path", config.get("model_path")),
            "model.path",
        )
        self.background_path = self._resolve_existing_file(
            self.background_config.get("path", config.get("background_path")),
            "background.path",
        )
        self.depth_model_type = str(
            self.depth_config.get("model_type", "DPT_Hybrid")
        )
        self.downsample_ratio = self._positive_number(
            self.model_config.get("downsample_ratio", 1.0),
            "model.downsample_ratio",
        )
        if self.downsample_ratio > 1.0:
            raise ValueError("model.downsample_ratio must be in the range (0, 1].")
        self.harmonize_colors = bool(
            self.enhancement_config.get("color_harmonization", True)
        )
        self.status_text = str(config.get("status_text", "Dual feeds active"))
        self.use_cuda_streams = bool(
            config.get("use_cuda_streams", self.device.type == "cuda")
        )

        self.capture = None
        self.matting = None
        self.harmonizer = None
        self.depth = None
        self.composer = None
        self.cuda_streams = None
        self.fps_counter = FPSCounter(
            window_size=int(self.config.get("fps_window_size", 30))
        )
        self.overlay = OverlayRenderer()
        self._cleaned_up = False

        try:
            if self.device.type == "cuda" and self.use_cuda_streams:
                self.cuda_streams = CudaStreamManager(self.device)
            self.capture = DualStreamCapture(
                cam_indices=self.camera_indices,
                resolution=self.resolution,
                fps=self.fps,
            )
            if self.capture.stream_a is None or self.capture.stream_b is None:
                raise RuntimeError(
                    "Could not open both configured camera devices "
                    f"{self.camera_indices}."
                )
            self.matting = MattingEngine(
                str(self.model_path),
                device=str(self.device),
                downsample_ratio=self.downsample_ratio,
            )
            self.harmonizer = ColorHarmonizer()
            self.depth = DepthEstimator(
                model_type=self.depth_model_type,
                device=str(self.device),
            )
            self.composer = CanvasComposer(
                str(self.background_path),
                self.resolution,
            )
        except Exception:
            self.cleanup()
            raise

    def _section(self, name: str) -> dict[str, Any]:
        value = self.config.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"Configuration section '{name}' must be a mapping.")
        return value

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.project_dir / path
        return path.resolve()

    def _resolve_existing_file(self, value: Any, name: str) -> Path:
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"Configuration must define {name} as a file path.")
        path = self._resolve_path(value)
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
        return path

    @staticmethod
    def _positive_number(value: Any, name: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive number.") from exc
        if not np.isfinite(number) or number <= 0:
            raise ValueError(f"{name} must be a positive number.")
        return number

    def _get_camera_indices(self) -> tuple[int, int]:
        indices = self.camera_config.get(
            "indices",
            self.config.get("camera_indices", (0, 1)),
        )
        if (
            not isinstance(indices, (list, tuple))
            or len(indices) != 2
            or any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
            or indices[0] == indices[1]
        ):
            raise ValueError("camera.indices must contain two distinct integer device indices.")
        return indices[0], indices[1]

    def _get_resolution(self) -> tuple[int, int]:
        resolution = self.camera_config.get(
            "resolution",
            self.config.get("resolution", (1280, 720)),
        )
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
            raise ValueError("camera.resolution must be [width, height].")
        try:
            width, height = (int(dimension) for dimension in resolution)
        except (TypeError, ValueError) as exc:
            raise ValueError("camera.resolution dimensions must be integers.") from exc
        if width < 1 or height < 1:
            raise ValueError("camera.resolution dimensions must be positive.")
        return width, height

    @staticmethod
    def _select_device(device_name: str) -> torch.device:
        device_name = device_name.lower()
        if device_name.startswith("cuda"):
            if torch.cuda.is_available():
                device = torch.device(device_name)
                if device.index is not None and device.index >= torch.cuda.device_count():
                    raise ValueError(f"Requested CUDA device does not exist: {device_name}")
                return device
            LOGGER.warning("CUDA is unavailable; falling back to CPU.")
            return torch.device("cpu")
        if device_name == "cpu":
            return torch.device("cpu")
        raise ValueError("device must be 'cpu' or a CUDA device such as 'cuda:0'.")

    def _configure_cuda_memory(self) -> None:
        if self.device.type != "cuda":
            return
        memory_fraction = self.config.get("cuda_memory_fraction")
        if memory_fraction is not None:
            fraction = self._positive_number(memory_fraction, "cuda_memory_fraction")
            if fraction > 1.0:
                raise ValueError("cuda_memory_fraction cannot exceed 1.0.")
            torch.cuda.set_per_process_memory_fraction(fraction, self.device)

    def _wait_for_quit(self) -> bool:
        return cv2.waitKey(1) & 0xFF == ord("q")

    def _resize_predictions(
        self,
        alpha: np.ndarray,
        foreground: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        width, height = self.resolution
        if alpha.shape != (height, width):
            alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)
        if foreground.shape[:2] != (height, width):
            foreground = cv2.resize(
                foreground,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
        return np.clip(alpha, 0.0, 1.0), foreground

    def _process_frames(self, frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
        width, height = self.resolution
        if frame_a.shape[:2] != (height, width):
            frame_a = cv2.resize(frame_a, (width, height), interpolation=cv2.INTER_LINEAR)
        if frame_b.shape[:2] != (height, width):
            frame_b = cv2.resize(frame_b, (width, height), interpolation=cv2.INTER_LINEAR)

        tensor_batch = self.matting.preprocess(frame_a, frame_b)
        alpha_raw, foreground_raw = self.matting.predict_alpha_batch(tensor_batch)
        if self.cuda_streams is not None:
            self.cuda_streams.synchronize()
        alpha_a, alpha_b, foreground_a, foreground_b = self.matting.postprocess(
            alpha_raw,
            foreground_raw,
        )
        alpha_a, foreground_a = self._resize_predictions(alpha_a, foreground_a)
        alpha_b, foreground_b = self._resize_predictions(alpha_b, foreground_b)

        depth_a = self.depth.estimate_depth_map(frame_a)
        depth_b = self.depth.estimate_depth_map(frame_b)
        z_order = self.depth.get_z_order(depth_a, alpha_a, depth_b, alpha_b)

        if self.harmonize_colors:
            foreground_a, foreground_b = self.harmonizer.harmonize_pair(
                foreground_a,
                alpha_a,
                foreground_b,
                alpha_b,
            )

        canvas = self.composer.composite_dual_subjects(
            foreground_a,
            alpha_a,
            foreground_b,
            alpha_b,
            z_order,
        )
        self.fps_counter.update()
        return self.overlay.draw_hud(
            canvas,
            self.fps_counter.get_fps(),
            z_order,
            self.status_text,
        )

    def run(self) -> None:
        try:
            while True:
                try:
                    frame_a, frame_b = self.capture.get_synced_frames()
                    if frame_a is None or frame_b is None:
                        LOGGER.warning("One or both camera feeds have no frame yet.")
                        if self._wait_for_quit():
                            break
                        continue

                    final_canvas = self._process_frames(frame_a, frame_b)
                    cv2.imshow("Photobooth", final_canvas)
                except Exception:
                    LOGGER.exception("Failed to process a frame; skipping and recovering.")

                if self._wait_for_quit():
                    break
        except KeyboardInterrupt:
            LOGGER.info("Photobooth interrupted by user.")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        if self._cleaned_up:
            return
        self._cleaned_up = True
        if self.capture is not None:
            self.capture.stop()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cv2.destroyAllWindows()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run the dual-subject virtual photobooth.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    try:
        PhotoboothApp(args.config).run()
    except Exception:
        LOGGER.exception("Could not start the photobooth.")
        raise


if __name__ == "__main__":
    main()
