import cv2
import numpy as np


class CanvasComposer:
    """Blend two alpha-matted subjects onto a virtual background."""

    def __init__(self, bg_path, target_resolution):
        self.target_resolution = tuple(target_resolution)
        bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)
        if bg is None:
            raise FileNotFoundError(f"Background image not found: {bg_path}")

        bg = cv2.resize(bg, (self.target_resolution[0], self.target_resolution[1]), interpolation=cv2.INTER_LINEAR)
        self.bg = bg.astype(np.float32)

    def _prepare_alpha(self, alpha, shape):
        alpha = np.asarray(alpha, dtype=np.float32)
        if alpha.shape == shape[:2]:
            return alpha[..., None]
        if alpha.shape == shape:
            return alpha
        if alpha.shape == (shape[0], shape[1], 1):
            return alpha
        raise ValueError(f"Alpha mask shape {alpha.shape} is not broadcastable to {shape}.")

    def _prepare_image(self, image, shape):
        image = np.asarray(image, dtype=np.float32)
        if image.shape != shape:
            if image.shape[:2] == shape[:2] and image.shape[2] == 3:
                return image.astype(np.float32)
            if image.shape == (shape[0], shape[1], 1):
                return np.repeat(image, 3, axis=2)
            raise ValueError(f"Foreground image shape {image.shape} does not match target {shape}.")
        return image.astype(np.float32)

    def composite_dual_subjects(self, fgr_A, alpha_A, fgr_B, alpha_B, z_order):
        if fgr_A is None or fgr_B is None:
            raise ValueError("Both foreground images must be provided.")

        bg_shape = self.bg.shape
        fgr_A = self._prepare_image(fgr_A, bg_shape)
        fgr_B = self._prepare_image(fgr_B, bg_shape)
        alpha_A = self._prepare_alpha(alpha_A, bg_shape)
        alpha_B = self._prepare_alpha(alpha_B, bg_shape)

        alpha_A = np.clip(alpha_A, 0.0, 1.0)
        alpha_B = np.clip(alpha_B, 0.0, 1.0)

        if z_order == "A_over_B":
            mid = alpha_B * fgr_B + (1.0 - alpha_B) * self.bg
            final = alpha_A * fgr_A + (1.0 - alpha_A) * mid
        elif z_order == "B_over_A":
            mid = alpha_A * fgr_A + (1.0 - alpha_A) * self.bg
            final = alpha_B * fgr_B + (1.0 - alpha_B) * mid
        else:
            raise ValueError(f"Unsupported z_order '{z_order}'. Expected 'A_over_B' or 'B_over_A'.")

        final = np.clip(final, 0.0, 255.0)
        final = np.rint(final).astype(np.uint8)
        return final


__all__ = ["CanvasComposer"]
