"""Local browser test server for the dual-camera matting/compositing pipeline.

Example: python local_test_backend.py --input-a 0 --input-b 1 --model models/weights/model.pth --background data/raw/background.jpg
Video files can be used in place of camera indices. Use --synthetic to check the UI without weights/cameras.
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Local CV pipeline test")


class LocalPipeline:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.frames = [None, None]
        self.output = self._placeholder()
        self.fps = 0.0
        self.drops = 0
        self.error = ""
        self.captures = []
        self.readers = []
        self.composer = None
        self.engine = None
        self.thread = None
        if not args.synthetic:
            self.captures = [self._open(args.input_a), self._open(args.input_b)]
            for cap in self.captures:
                if not cap.isOpened():
                    raise RuntimeError("Could not open an input. Use camera indices or valid video paths.")
            if not args.model or not Path(args.model).is_file():
                raise FileNotFoundError("Provide a trained .pth checkpoint with --model.")
            if not args.background or not Path(args.background).is_file():
                raise FileNotFoundError("Provide a background image with --background.")
            from src.segmentation.inference import MattingEngine
            from src.compositing.alpha_blender import CanvasComposer
            torch.set_num_threads(max(1, args.threads))
            self.engine = MattingEngine(args.model, device="cpu", downsample_ratio=args.scale)
            self.composer = CanvasComposer(args.background, (args.width, args.height))
            self.readers = [threading.Thread(target=self._read_loop, args=(i,), daemon=True) for i in range(2)]
            for reader in self.readers:
                reader.start()
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    @staticmethod
    def _open(source):
        try:
            src = int(source)
        except ValueError:
            src = source
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _placeholder(self):
        im = np.zeros((self.args.height, self.args.width, 3), np.uint8)
        cv2.putText(im, "Waiting for pipeline", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)
        return im

    def _read_loop(self, index):
        while not self.stop_event.is_set():
            ok, frame = self.captures[index].read()
            if not ok:
                # Loop video files to keep a continuous feed; cameras retry.
                if isinstance(self.captures[index].get(cv2.CAP_PROP_POS_FRAMES), float) and self.captures[index].get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                    self.captures[index].set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    time.sleep(.02)
                continue
            frame = cv2.resize(frame, (self.args.width, self.args.height), interpolation=cv2.INTER_AREA)
            with self.lock:
                if self.frames[index] is not None:
                    self.drops += 1
                self.frames[index] = frame

    def _process_loop(self):
        interval = 1.0 / max(0.1, self.args.target_fps)
        prior = time.perf_counter()
        count = 0
        while not self.stop_event.is_set():
            started = time.perf_counter()
            with self.lock:
                pair = [None if f is None else f.copy() for f in self.frames]
            if self.args.synthetic:
                pair = [self._synthetic(0), self._synthetic(1)]
                canvas = np.zeros((self.args.height, self.args.width, 3), np.uint8)
                canvas[:] = (55, 75, 95)
                for i, frame in enumerate(pair):
                    cx = int((time.time() * 70 + i * self.args.width / 2) % self.args.width)
                    cv2.circle(frame, (cx, self.args.height // 2), max(20, self.args.height // 5), (40, 180, 240), -1)
                    cv2.circle(canvas, (cx, self.args.height // 2), max(20, self.args.height // 5), (40, 180, 240), -1)
                with self.lock:
                    self.frames = pair
            elif all(f is not None for f in pair):
                try:
                    batch = self.engine.preprocess(pair[0], pair[1])
                    with torch.inference_mode():
                        alpha_t, _ = self.engine.predict_alpha_batch(batch)
                    alpha_a, alpha_b = [np.clip(x, 0, 1).astype(np.float32) for x in alpha_t[:, 0].detach().cpu().numpy()]
                    alpha_a = cv2.resize(alpha_a, (self.args.width, self.args.height))
                    alpha_b = cv2.resize(alpha_b, (self.args.width, self.args.height))
                    # MattingModelBuilder currently predicts alpha only, so retain captured RGB as foreground.
                    canvas = self.composer.composite_dual_subjects(pair[0], alpha_a, pair[1], alpha_b, "A_over_B")
                except Exception as exc:
                    self.error = str(exc)
                    canvas = self._placeholder()
            else:
                canvas = self._placeholder()
            with self.lock:
                self.output = canvas
            count += 1
            now = time.perf_counter()
            if now - prior >= 1:
                self.fps = count / (now - prior)
                count, prior = 0, now
            time.sleep(max(0, interval - (time.perf_counter() - started)))

    def _synthetic(self, index):
        h, w = self.args.height, self.args.width
        im = np.zeros((h, w, 3), np.uint8)
        im[:] = (70 + index * 35, 50, 30)
        cv2.putText(im, f"Synthetic Feed {index+1}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)
        return im

    def jpeg(self, which):
        with self.lock:
            if which == "output": frame = self.output.copy()
            else:
                frame = self.frames[int(which)]
                frame = self._placeholder() if frame is None else frame.copy()
        return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

    def close(self):
        self.stop_event.set()
        for cap in self.captures:
            cap.release()


@app.on_event("startup")
def startup():
    if not hasattr(app.state, "pipeline"):
        app.state.pipeline = LocalPipeline(_ARGS)


@app.on_event("shutdown")
def shutdown():
    if hasattr(app.state, "pipeline"):
        app.state.pipeline.close()


@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "web" / "index.html").read_text(encoding="utf-8")


@app.get("/feed/{name}")
def feed(name: str):
    if name not in ("a", "b", "output"):
        return JSONResponse({"error": "feed name must be a, b, or output"}, status_code=404)
    p = app.state.pipeline
    def stream():
        while not p.stop_event.is_set():
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + p.jpeg(name) + b"\r\n"
            time.sleep(.04)
    return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/stats")
def stats():
    p = app.state.pipeline
    return {"fps": round(p.fps, 2), "drops": p.drops, "mode": "synthetic" if p.args.synthetic else "cpu-model", "error": p.error}


_ARGS = None
def main():
    global _ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-a", default="0", help="camera index or video file")
    parser.add_argument("--input-b", default="1", help="camera index or video file")
    parser.add_argument("--model", help="trained .pth checkpoint")
    parser.add_argument("--background", help="background image")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--scale", type=float, default=.5, help="model resolution scale, 0-1")
    parser.add_argument("--target-fps", type=float, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--synthetic", action="store_true", help="UI smoke mode, no camera/model required")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    _ARGS = parser.parse_args()
    if not (0 < _ARGS.scale <= 1) or min(_ARGS.width, _ARGS.height, _ARGS.threads) <= 0:
        parser.error("scale must be in (0,1], dimensions and threads must be positive")
    uvicorn.run(app, host=_ARGS.host, port=_ARGS.port)


if __name__ == "__main__":
    main()
