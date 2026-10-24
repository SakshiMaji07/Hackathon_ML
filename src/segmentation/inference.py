from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from src.segmentation.rvm_model import MattingModelBuilder


class MattingEngine:
    """Real-time batched human matting engine with temporal recurrence."""

    def __init__(self, model_path, device="cuda", downsample_ratio=1.0):
        self.model_path = model_path
        self.device = device
        self.downsample_ratio = float(downsample_ratio)
        if not (0.0 < self.downsample_ratio <= 1.0):
            raise ValueError("downsample_ratio must be in the range (0, 1].")

        builder = MattingModelBuilder(backbone="mobilenet_v3_small", half_precision=(device.lower() == "cuda"))
        self.model = builder.load_rvm_backbone(model_path, device=device)

        self.r1 = None
        self.r2 = None
        self.r3 = None
        self.r4 = None

    def _ensure_recurrent_states(self, batch_tensor):
        if batch_tensor.dim() != 4 or batch_tensor.shape[0] != 2:
            raise ValueError(f"Expected a batch tensor of shape (2, C, H, W), got {tuple(batch_tensor.shape)}")

        if self.r1 is not None:
            return

        b, _, h, w = batch_tensor.shape
        state_shapes = [
            (b, 16, max(1, h // 8), max(1, w // 8)),
            (b, 16, max(1, h // 8), max(1, w // 8)),
            (b, 32, max(1, h // 16), max(1, w // 16)),
            (b, 32, max(1, h // 16), max(1, w // 16)),
        ]
        self.r1 = torch.zeros(state_shapes[0], device=batch_tensor.device, dtype=batch_tensor.dtype)
        self.r2 = torch.zeros(state_shapes[1], device=batch_tensor.device, dtype=batch_tensor.dtype)
        self.r3 = torch.zeros(state_shapes[2], device=batch_tensor.device, dtype=batch_tensor.dtype)
        self.r4 = torch.zeros(state_shapes[3], device=batch_tensor.device, dtype=batch_tensor.dtype)

    def preprocess(self, frame_A, frame_B):
        if frame_A is None or frame_B is None:
            raise ValueError("Both camera frames must be valid BGR arrays.")
        if frame_A.shape != frame_B.shape:
            raise ValueError(f"Frame shapes must match for batch inference: {frame_A.shape} vs {frame_B.shape}")

        processed = []
        for frame in (frame_A, frame_B):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous()
            processed.append(tensor)

        batch = torch.stack(processed, dim=0).to(self.device)
        if self.downsample_ratio != 1.0:
            batch = F.interpolate(batch, scale_factor=self.downsample_ratio, mode="bilinear", align_corners=False)
        return batch

    def _parse_model_output(self, output):
        alpha = None
        fgr = None
        states = [None, None, None, None]

        if isinstance(output, dict):
            for key in ["alpha", "alpha_map", "pred_alpha", "mask"]:
                if key in output:
                    alpha = output[key]
                    break
            for key in ["fgr", "foreground", "fgr_map", "rgb"]:
                if key in output:
                    fgr = output[key]
                    break
            for idx, key in enumerate(["r1", "r2", "r3", "r4"]):
                if key in output:
                    states[idx] = output[key]
            return alpha, fgr, states

        if isinstance(output, (tuple, list)):
            if len(output) >= 2:
                alpha, fgr = output[0], output[1]
            for idx in range(2, min(len(output), 6)):
                if idx - 2 < 4:
                    states[idx - 2] = output[idx]
            return alpha, fgr, states

        alpha = output
        return alpha, fgr, states

    def predict_alpha_batch(self, tensor_batch):
        if tensor_batch.dim() != 4 or tensor_batch.shape[0] != 2:
            raise ValueError(f"Expected batch tensor of shape (2, 3, H, W), got {tuple(tensor_batch.shape)}")

        self._ensure_recurrent_states(tensor_batch)

        with torch.no_grad():
            try:
                raw_output = self.model(tensor_batch, self.r1, self.r2, self.r3, self.r4)
            except TypeError:
                try:
                    raw_output = self.model(tensor_batch, self.r1, self.r2, self.r3, self.r4, self.downsample_ratio)
                except TypeError:
                    raw_output = self.model(tensor_batch)

            alpha, fgr, new_states = self._parse_model_output(raw_output)

            if alpha is None:
                raise ValueError("Model output did not include alpha values.")
            if alpha.ndim != 4 or alpha.shape[1] not in (1, 3):
                raise ValueError(
                    f"Expected model alpha output with shape (B, 1, H, W) or "
                    f"(B, 3, H, W), got {tuple(alpha.shape)}."
                )
            if alpha.shape[-2:] != tensor_batch.shape[-2:]:
                alpha = F.interpolate(
                    alpha,
                    size=tensor_batch.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            alpha = torch.sigmoid(alpha)
            if fgr is None:
                fgr = torch.ones_like(alpha.expand(-1, 3, -1, -1), device=alpha.device)
            elif fgr.shape[-2:] != tensor_batch.shape[-2:]:
                fgr = F.interpolate(
                    fgr,
                    size=tensor_batch.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            for idx, state in enumerate(new_states):
                if state is not None:
                    setattr(self, f"r{idx + 1}", state)

            return alpha, fgr

    def postprocess(self, alpha_tensor, fgr_tensor):
        if alpha_tensor is None:
            raise ValueError("Alpha tensor is missing.")

        alpha_np = alpha_tensor.detach().cpu().float().numpy()
        fgr_np = fgr_tensor.detach().cpu().float().numpy()

        if alpha_np.ndim == 4 and alpha_np.shape[1] == 1:
            alpha_np = alpha_np[:, 0]
        if alpha_np.ndim == 4 and alpha_np.shape[1] == 3:
            alpha_np = alpha_np[:, 0]

        alpha_a = np.clip(alpha_np[0], 0.0, 1.0).astype(np.float32)
        alpha_b = np.clip(alpha_np[1], 0.0, 1.0).astype(np.float32)

        if fgr_np.ndim == 4:
            fgr_np = fgr_np.permute(0, 2, 3, 1) if hasattr(fgr_np, "permute") else fgr_np.transpose(0, 2, 3, 1)

        fgr_a = np.clip(fgr_np[0][..., ::-1] * 255.0, 0, 255).astype(np.uint8)
        fgr_b = np.clip(fgr_np[1][..., ::-1] * 255.0, 0, 255).astype(np.uint8)

        return alpha_a, alpha_b, fgr_a, fgr_b

    def predict(self, frame_A, frame_B):
        batch_tensor = self.preprocess(frame_A, frame_B)
        alpha_batch, fgr_batch = self.predict_alpha_batch(batch_tensor)
        return self.postprocess(alpha_batch, fgr_batch)


__all__ = ["MattingEngine"]
