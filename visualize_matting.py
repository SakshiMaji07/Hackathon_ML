from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.segmentation.inference import MattingEngine


def _open_video(path: Path, label: str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open {label} video: {path}")
    return capture


def _read_looping_frame(capture: cv2.VideoCapture, path: Path) -> np.ndarray:
    ok, frame = capture.read()
    if not ok or frame is None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = capture.read()
    if not ok or frame is None:
        raise OSError(f"Could not read a frame from video: {path}")
    return frame


def _composite_pair(
    frame_a: np.ndarray,
    alpha_a: np.ndarray,
    frame_b: np.ndarray,
    alpha_b: np.ndarray,
    background: np.ndarray,
    z_order: str,
) -> np.ndarray:
    foreground_a = frame_a.astype(np.float32)
    foreground_b = frame_b.astype(np.float32)
    alpha_a = np.clip(alpha_a, 0.0, 1.0)[..., None].astype(np.float32)
    alpha_b = np.clip(alpha_b, 0.0, 1.0)[..., None].astype(np.float32)
    background = background.astype(np.float32)

    if z_order == "A_over_B":
        underlay = alpha_b * foreground_b + (1.0 - alpha_b) * background
        output = alpha_a * foreground_a + (1.0 - alpha_a) * underlay
    else:
        underlay = alpha_a * foreground_a + (1.0 - alpha_a) * background
        output = alpha_b * foreground_b + (1.0 - alpha_b) * underlay
    return np.clip(output, 0.0, 255.0).astype(np.uint8)


def run(args: argparse.Namespace) -> None:
    video_a_path = Path(args.video_a).expanduser().resolve()
    video_b_path = Path(args.video_b).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    for label, path in (
        ("Video A", video_a_path),
        ("Video B", video_b_path),
        ("Model checkpoint", model_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    if args.width < 1 or args.height < 1:
        raise ValueError("Output width and height must be positive.")
    if not 0.0 < args.scale <= 1.0:
        raise ValueError("Model scale must be in the range (0, 1].")
    if args.threads < 1:
        raise ValueError("CPU thread count must be positive.")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.")
        device = "cpu"
    else:
        device = args.device
    torch.set_num_threads(args.threads)

    capture_a = _open_video(video_a_path, "Video A")
    try:
        capture_b = _open_video(video_b_path, "Video B")
    except Exception:
        capture_a.release()
        raise

    try:
        engine = MattingEngine(
            str(model_path),
            device=device,
            downsample_ratio=args.scale,
            backbone=args.backbone,
        )
        width, height = args.width, args.height
        background = np.full(
            (height, width, 3),
            args.background_color,
            dtype=np.uint8,
        )

        cv2.namedWindow("Video A", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Video B", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Segmented Output", cv2.WINDOW_NORMAL)

        while True:
            frame_a = _read_looping_frame(capture_a, video_a_path)
            frame_b = _read_looping_frame(capture_b, video_b_path)
            frame_a = cv2.resize(frame_a, (width, height), interpolation=cv2.INTER_AREA)
            frame_b = cv2.resize(frame_b, (width, height), interpolation=cv2.INTER_AREA)

            tensor_batch = engine.preprocess(frame_a, frame_b)
            alpha_batch, _ = engine.predict_alpha_batch(tensor_batch)
            alpha_a, alpha_b = alpha_batch[:, 0].detach().cpu().numpy()
            if alpha_a.shape != (height, width):
                alpha_a = cv2.resize(alpha_a, (width, height), interpolation=cv2.INTER_LINEAR)
                alpha_b = cv2.resize(alpha_b, (width, height), interpolation=cv2.INTER_LINEAR)

            output = _composite_pair(
                frame_a,
                alpha_a,
                frame_b,
                alpha_b,
                background,
                args.z_order,
            )
            cv2.imshow("Video A", frame_a)
            cv2.imshow("Video B", frame_b)
            cv2.imshow("Segmented Output", output)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture_a.release()
        capture_b.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run matting inference on two videos in three OpenCV windows."
    )
    parser.add_argument("--video-a", required=True, help="Path to the first input video.")
    parser.add_argument("--video-b", required=True, help="Path to the second input video.")
    parser.add_argument(
        "--model",
        default="models/weights/best_model.pth",
        help="Checkpoint compatible with the repository's MobileNetV3 alpha model.",
    )
    parser.add_argument(
        "--backbone",
        choices=("mobilenet_v3_small", "resnet50"),
        default="mobilenet_v3_small",
        help="Backbone used to train the checkpoint.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device; auto selects CUDA when available, otherwise CPU.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--background-color",
        type=int,
        nargs=3,
        metavar=("B", "G", "R"),
        default=(48, 48, 48),
        help="Solid BGR background color used for compositing the segmented people.",
    )
    parser.add_argument(
        "--z-order",
        choices=("A_over_B", "B_over_A"),
        default="A_over_B",
        help="Which segmented video is composited in front.",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
