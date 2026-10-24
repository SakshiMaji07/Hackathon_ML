import numpy as np
import cv2


class ColorHarmonizer:
    """Match lighting/color between two foreground subjects using LAB statistics."""

    @staticmethod
    def _safe_std(values):
        std = np.std(values)
        if np.isclose(std, 0.0):
            return 1.0
        return std

    @staticmethod
    def extract_lab_stats(fgr_image, alpha_mask):
        if fgr_image is None:
            raise ValueError("Foreground image cannot be None.")
        if alpha_mask is None:
            raise ValueError("Alpha mask cannot be None.")

        fgr_image = np.asarray(fgr_image)
        alpha_mask = np.asarray(alpha_mask, dtype=np.float32)

        if fgr_image.ndim != 3 or fgr_image.shape[2] != 3:
            raise ValueError(f"Expected BGR image with shape (H, W, 3), got {fgr_image.shape}.")
        if alpha_mask.shape[:2] != fgr_image.shape[:2]:
            raise ValueError(f"Alpha mask shape {alpha_mask.shape} does not match image shape {fgr_image.shape[:2]}.")

        valid_mask = alpha_mask > 0.0
        if not np.any(valid_mask):
            return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)

        lab_image = cv2.cvtColor(fgr_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        masked = lab_image[valid_mask]
        mean = masked.mean(axis=0)
        std = masked.std(axis=0)
        std = np.where(np.isclose(std, 0.0), 1.0, std)
        return mean.astype(np.float32), std.astype(np.float32)

    @staticmethod
    def apply_color_transfer(source_fgr, source_alpha, target_fgr, target_alpha):
        if source_fgr is None or target_fgr is None:
            raise ValueError("Both source and target foreground images are required.")

        source_fgr = np.asarray(source_fgr, dtype=np.uint8)
        target_fgr = np.asarray(target_fgr, dtype=np.uint8)
        source_alpha = np.asarray(source_alpha, dtype=np.float32)
        target_alpha = np.asarray(target_alpha, dtype=np.float32)

        if source_fgr.shape != target_fgr.shape:
            raise ValueError(f"Foreground shapes must match: {source_fgr.shape} vs {target_fgr.shape}")
        if source_alpha.shape[:2] != source_fgr.shape[:2] or target_alpha.shape[:2] != target_fgr.shape[:2]:
            raise ValueError("Alpha masks must match foreground dimensions.")

        src_mean, src_std = ColorHarmonizer.extract_lab_stats(source_fgr, source_alpha)
        tgt_mean, tgt_std = ColorHarmonizer.extract_lab_stats(target_fgr, target_alpha)

        source_lab = cv2.cvtColor(source_fgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        source_lab = source_lab.copy()

        for channel_idx in range(3):
            if np.isclose(src_std[channel_idx], 0.0):
                src_std[channel_idx] = 1.0
            if np.isclose(tgt_std[channel_idx], 0.0):
                tgt_std[channel_idx] = 1.0

            source_lab[:, :, channel_idx] = ((source_lab[:, :, channel_idx] - src_mean[channel_idx]) / src_std[channel_idx]) * tgt_std[channel_idx] + tgt_mean[channel_idx]

        source_lab = np.clip(source_lab, [0, -127, -127], [100, 127, 127])
        adjusted_bgr = cv2.cvtColor(source_lab.astype(np.float32), cv2.COLOR_LAB2BGR)
        adjusted_bgr = np.clip(adjusted_bgr, 0, 255).astype(np.uint8)

        alpha_mask = (source_alpha > 0).astype(np.uint8)
        adjusted = source_fgr.copy()
        adjusted[alpha_mask == 1] = adjusted_bgr[alpha_mask == 1]

        return adjusted

    def harmonize_pair(self, fgr_A, alpha_A, fgr_B, alpha_B):
        fgr_a_adj = self.apply_color_transfer(fgr_A, alpha_A, fgr_B, alpha_B)
        fgr_b_adj = self.apply_color_transfer(fgr_B, alpha_B, fgr_A, alpha_A)
        return fgr_a_adj, fgr_b_adj


__all__ = ["ColorHarmonizer"]
