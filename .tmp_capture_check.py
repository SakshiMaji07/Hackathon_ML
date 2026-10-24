import time
import cv2
import numpy as np
from src.input.stream_capture import DualStreamCapture, WebcamStream

class FakeCapture:
    def __init__(self, index):
        self.index = index
    def isOpened(self):
        return True
    def set(self, prop, value):
        return True
    def read(self):
        frame = np.full((60, 80, 3), self.index * 50, dtype=np.uint8)
        frame[:, :, 1] = 100
        frame[:, :, 2] = 200
        return True, frame
    def release(self):
        pass

cv2.VideoCapture = lambda index: FakeCapture(index)
stream = WebcamStream(0, (80, 60), 15)
time.sleep(0.1)
frame = stream.read()
assert isinstance(frame, np.ndarray)
assert frame.shape == (60, 80, 3)
assert frame.dtype == np.uint8
assert (frame[:, :, 0] == 0).all()
stream.stop()

cap = DualStreamCapture((0, 1), (80, 60), 15)
time.sleep(0.1)
fa, fb = cap.get_synced_frames()
assert fa.shape == (60, 80, 3) and fb.shape == (60, 80, 3)
cap.stop()
print('capture validation passed')
