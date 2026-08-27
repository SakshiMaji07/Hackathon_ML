# Sequential Pipeline Order
from src.01_input.stream_capture import DualStreamCapture
from src.02_segmentation.inference import MattingEngine
from src.03_enhancement.color_harmonizer import Harmonizer
from src.03_enhancement.depth_estimation import DepthEstimator
from src.04_compositing.alpha_blender import CanvasComposer

# 1. Grab async frames from both feeds
# 2. Extract alpha matte for Subject A and Subject B
# 3. Apply color matching and check z-depth layering
# 4. Blend both subjects onto final canvas and display