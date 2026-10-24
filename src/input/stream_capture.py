import queue
import threading

import cv2
import numpy as np


class WebcamStream:
    """Background camera reader that keeps only the most recent frame in a 1-slot queue."""

    def __init__(self, cam_index, resolution, fps):
        self.cam_index = cam_index
        self.resolution = tuple(resolution)
        self.fps = fps

        self.running = False
        self.thread = None
        self.latest_frame = None
        self.frame_queue = queue.Queue(maxsize=1)
        self._latest_lock = threading.Lock()

        self.capture = cv2.VideoCapture(cam_index)
        if self.capture is None or not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera index {cam_index}.")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.resolution[0]))
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.resolution[1]))
        self.capture.set(cv2.CAP_PROP_FPS, float(self.fps))

        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            if self.capture is None or not self.capture.isOpened():
                break

            ret, frame = self.capture.read()
            if not ret or frame is None:
                continue

            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass

    def read(self):
        try:
            frame = self.frame_queue.get_nowait()
        except queue.Empty:
            return self.latest_frame

        if frame is not None:
            with self._latest_lock:
                self.latest_frame = frame
        return self.latest_frame

    def stop(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class DualStreamCapture:
    """Reader that manages two independent camera streams for a dual-camera pipeline."""

    def __init__(self, cam_indices=(0, 1), resolution=(1280, 720), fps=30):
        self.resolution = tuple(resolution)
        self.fps = fps

        self.stream_a = self._create_stream(cam_indices[0])
        self.stream_b = self._create_stream(cam_indices[1])

    def _create_stream(self, cam_index):
        try:
            return WebcamStream(cam_index=cam_index, resolution=self.resolution, fps=self.fps)
        except RuntimeError as exc:
            print(f"Warning: {exc}")
            return None

    def get_synced_frames(self):
        frame_a = self.stream_a.read() if self.stream_a is not None else None
        frame_b = self.stream_b.read() if self.stream_b is not None else None
        return frame_a, frame_b

    def stop(self):
        if getattr(self, "stream_a", None) is not None:
            self.stream_a.stop()
        if getattr(self, "stream_b", None) is not None:
            self.stream_b.stop()


if __name__ == "__main__":
    capture = DualStreamCapture(cam_indices=(0, 1), resolution=(1280, 720), fps=30)
    try:
        while True:
            frame_a, frame_b = capture.get_synced_frames()
            if frame_a is not None:
                cv2.imshow("Camera A", frame_a)
            if frame_b is not None:
                cv2.imshow("Camera B", frame_b)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.stop()
        cv2.destroyAllWindows()
