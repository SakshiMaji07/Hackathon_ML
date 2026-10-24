from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MattingLoss(nn.Module):
    """Combined alpha, composite-fidelity, and spatial-gradient loss."""

    def __init__(self, w1: float = 1.0, w2: float = 1.0, w3: float = 1.0):
        super().__init__()
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.w3 = float(w3)
        self.eps = 1e-6

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        )
        sobel_y = sobel_x.T
        self.register_buffer("sobel_x", sobel_x.reshape(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.reshape(1, 1, 3, 3))

    def alpha_loss(self, pred_alpha: torch.Tensor, gt_alpha: torch.Tensor) -> torch.Tensor:
        difference = pred_alpha - gt_alpha
        l1 = difference.abs().mean()
        charbonnier = torch.sqrt(difference.square() + self.eps**2).mean()
        return l1 + charbonnier

    def comp_loss(
        self,
        pred_fg: torch.Tensor,
        pred_alpha: torch.Tensor,
        bg: torch.Tensor,
        gt_img: torch.Tensor,
    ) -> torch.Tensor:
        synth_comp = pred_alpha * pred_fg + (1.0 - pred_alpha) * bg
        return F.l1_loss(synth_comp, gt_img)

    @staticmethod
    def _as_batched_alpha(alpha: torch.Tensor) -> torch.Tensor:
        if alpha.ndim == 2:
            return alpha.unsqueeze(0).unsqueeze(0)
        if alpha.ndim == 3:
            return alpha.unsqueeze(1)
        if alpha.ndim == 4 and alpha.shape[1] == 1:
            return alpha
        raise ValueError(
            "Alpha tensors must have shape (H, W), (B, H, W), or (B, 1, H, W); "
            f"got {tuple(alpha.shape)}."
        )

    def spatial_gradient_loss(
        self,
        pred_alpha: torch.Tensor,
        gt_alpha: torch.Tensor,
    ) -> torch.Tensor:
        pred = self._as_batched_alpha(pred_alpha)
        target = self._as_batched_alpha(gt_alpha)
        if pred.shape != target.shape:
            raise ValueError(
                f"Predicted and ground-truth alpha shapes must match, got "
                f"{tuple(pred.shape)} and {tuple(target.shape)}."
            )

        sobel_x = self.sobel_x.to(dtype=pred.dtype)
        sobel_y = self.sobel_y.to(dtype=pred.dtype)
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)

        pred_grad = torch.cat((pred_grad_x, pred_grad_y), dim=1)
        target_grad = torch.cat((target_grad_x, target_grad_y), dim=1)
        return F.l1_loss(pred_grad, target_grad)

    def forward(self, pred_dict: dict[str, torch.Tensor], gt_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        alpha = pred_dict["alpha"]
        alpha_term = self.alpha_loss(alpha, gt_dict["alpha"])
        composite_term = self.comp_loss(
            pred_dict["fg"],
            alpha,
            gt_dict["bg"],
            gt_dict["composite"],
        )
        gradient_term = self.spatial_gradient_loss(alpha, gt_dict["alpha"])
        return self.w1 * alpha_term + self.w2 * composite_term + self.w3 * gradient_term