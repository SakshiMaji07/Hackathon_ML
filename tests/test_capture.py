import os
import time
from collections.abc import Iterator

import numpy as np
import pytest

from src.input.stream_capture import DualStreamCapture


def _camera_indices() -> tuple[int, int]:
    value = os.getenv("MATTE_CAMERA_INDICES", "0,1")
    try:
        indices = tuple(int(index.strip()) for index in value.split(","))
    except ValueError as exc:
        raise ValueError(
            "MATTE_CAMERA_INDICES must contain two comma-separated integers."
        ) from exc
    if len(indices) != 2:
        raise ValueError("MATTE_CAMERA_INDICES must contain exactly two camera indices.")
    return indices


def _resolution() -> tuple[int, int]:
    value = os.getenv("MATTE_CAMERA_RESOLUTION", "640,480")
    try:
        width, height = (int(dimension.strip()) for dimension in value.split(","))
    except ValueError as exc:
        raise ValueError(
            "MATTE_CAMERA_RESOLUTION must contain width,height as integers."
        ) from exc
    if width < 1 or height < 1:
        raise ValueError("Camera resolution dimensions must be positive.")
    return width, height


@pytest.fixture(scope="module")
def capture() -> Iterator[DualStreamCapture]:
    resolution = _resolution()
    stream_capture = DualStreamCapture(
        cam_indices=_camera_indices(),
        resolution=resolution,
        fps=30,
    )
    try:
        streams = (stream_capture.stream_a, stream_capture.stream_b)
        if any(
            stream is None
            or stream.capture is None
            or not stream.capture.isOpened()
            for stream in streams
        ):
            pytest.skip("Two physical camera devices are required for capture tests.")

        deadline = time.monotonic() + 5.0
        frames = (None, None)
        while time.monotonic() < deadline:
            frames = stream_capture.get_synced_frames()
            if all(frame is not None for frame in frames):
                break
            time.sleep(0.01)
        if any(frame is None for frame in frames):
            pytest.skip("Camera devices opened but did not provide frames.")
        yield stream_capture
    finally:
        stream_capture.stop()


class TestStreamCapture:
    def test_camera_connection(self, capture: DualStreamCapture) -> None:
        assert capture.stream_a is not None
        assert capture.stream_b is not None
        assert capture.stream_a.capture is not None
        assert capture.stream_b.capture is not None
        assert capture.stream_a.capture.isOpened()
        assert capture.stream_b.capture.isOpened()

    def test_frame_read_speed(self, capture: DualStreamCapture) -> None:
        resolution = _resolution()
        expected_shape = (resolution[1], resolution[0], 3)
        warmup_seconds = float(os.getenv("MATTE_CAPTURE_WARMUP_SECONDS", "0.5"))
        frame_count = int(os.getenv("MATTE_CAPTURE_TEST_FRAMES", "60"))
        minimum_fps = float(os.getenv("MATTE_CAPTURE_MIN_FPS", "30"))
        if warmup_seconds < 0 or frame_count < 1 or minimum_fps <= 0:
            raise ValueError(
                "Warmup must be non-negative, frame count and minimum FPS must be positive."
            )

        time.sleep(warmup_seconds)
        start = time.perf_counter()
        for _ in range(frame_count):
            frame_a, frame_b = capture.get_synced_frames()
            assert isinstance(frame_a, np.ndarray)
            assert isinstance(frame_b, np.ndarray)
            assert frame_a.shape == expected_shape
            assert frame_b.shape == expected_shape
        elapsed = time.perf_counter() - start
        achieved_fps = frame_count / elapsed if elapsed > 0 else float("inf")

        assert achieved_fps >= minimum_fps, (
            f"Read loop achieved {achieved_fps:.2f} FPS, below "
            f"the configured {minimum_fps:.2f} FPS threshold."
        )
