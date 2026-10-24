from __future__ import annotations

import numpy as np
import cv2
import torch


class DepthEstimator:
    """Monocular relative depth estimation using MiDaS for subject ordering."""

    def __init__(self, model_type="DPT_Hybrid", device="cuda"):
        self.model_type = model_type
        self.device = torch.device(device if device and device != "cuda" or torch.cuda.is_available() else "cpu")

        try:
            self.model = torch.hub.load(
                "intel-isl/MiDaS",
                model_type,
                pretrained=True,
                trust_repo=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load MiDaS model '{model_type}'.") from exc

        self.model.eval().to(self.device)

        if hasattr(self.model, "pretrained") and not self.model.pretrained:
            self.model.pretrained = True

        self.transforms = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms",
            trust_repo=True,
        )

        if model_type == "MiDaS_small":
            self.transform = self.transforms.small_transform
        else:
            self.transform = self.transforms.dpt_transform

    def estimate_depth_map(self, frame):
        if frame is None:
            raise ValueError("Input frame cannot be None.")

        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected BGR frame with shape (H, W, 3), got {frame.shape}.")

        with torch.no_grad():
            input_batch = self.transform(frame).to(self.device)
            prediction = self.model(input_batch)

            if isinstance(prediction, (tuple, list)):
                prediction = prediction[0]

            if prediction.dim() == 3:
                prediction = prediction.unsqueeze(0)

            depth_map = prediction.squeeze(0)
            depth_map = torch.nn.functional.interpolate(
                depth_map.unsqueeze(0).unsqueeze(0),
                size=(frame.shape[0], frame.shape[1]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

            depth_np = depth_map.cpu().numpy().astype(np.float32)
            return depth_np

    def get_z_order(self, depth_map_A, alpha_A, depth_map_B, alpha_B, threshold=0.05):
        if depth_map_A is None or depth_map_B is None:
            raise ValueError("Depth maps must not be None.")

        alpha_A = np.asarray(alpha_A, dtype=np.float32)
        alpha_B = np.asarray(alpha_B, dtype=np.float32)

        mask_A = alpha_A > threshold
        mask_B = alpha_B > threshold

        if mask_A.size == 0 or not np.any(mask_A):
            return "B_over_A"
        if mask_B.size == 0 or not np.any(mask_B):
            return "A_over_B"

        depth_A = np.asarray(depth_map_A, dtype=np.float32)
        depth_B = np.asarray(depth_map_B, dtype=np.float32)

        mean_depth_A = float(depth_A[mask_A].mean())
        mean_depth_B = float(depth_B[mask_B].mean())

        # MiDaS uses higher values as closer in the relative depth convention.
        if mean_depth_A > mean_depth_B:
            return "A_over_B"
        return "B_over_A"


__all__ = ["DepthEstimator"]
