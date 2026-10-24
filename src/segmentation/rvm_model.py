import os
from typing import Any, Dict, Optional, Union

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None

try:
    import torch
    import torch.nn as nn
    from torchvision import models as tv_models
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    tv_models = None


class _RVMEncoder(nn.Module):
    """Minimal RVM-style encoder with a MobileNetV3 or ResNet50 backbone."""

    def __init__(self, backbone_name: str = "mobilenet_v3_small"):
        super().__init__()
        backbone_name = backbone_name.lower()

        if backbone_name == "mobilenet_v3_small":
            base_model = tv_models.mobilenet_v3_small(weights=None)
            self.backbone = base_model.features
            self.neck = nn.Sequential(
                nn.Conv2d(576, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            self.output_dim = 128
        elif backbone_name == "resnet50":
            base_model = tv_models.resnet50(weights=None)
            self.backbone = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4,
            )
            self.neck = nn.Sequential(
                nn.Conv2d(2048, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            self.output_dim = 128
        else:
            raise ValueError(f"Unsupported backbone '{backbone_name}'. Use 'mobilenet_v3_small' or 'resnet50'.")

        self.alpha_head = nn.Conv2d(self.output_dim, 1, kernel_size=1)

    def forward(self, x):
        if hasattr(self, "backbone") and hasattr(self.backbone, "features"):
            feats = self.backbone(x)
        else:
            feats = self.backbone(x)

        fused = self.neck(feats)
        alpha = self.alpha_head(fused)
        return alpha


class MattingModelBuilder:
    """Factory for loading a Robust Video Matting backbone for inference."""

    def __init__(self, backbone: str = "mobilenet_v3_small", half_precision: bool = False):
        self.backbone = (backbone or "mobilenet_v3_small").lower()
        self.half_precision = bool(half_precision)

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        target = (device or "cpu").lower()
        if target == "cuda":
            if torch is not None and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        if target.startswith("cuda:"):
            if torch is not None and torch.cuda.is_available():
                return target
            return "cpu"
        return "cpu"

    def _build_rvm_model(self, backbone_name: str):
        if torch is None or nn is None or tv_models is None:
            raise ImportError("PyTorch and TorchVision are required to load .pth weights.")

        return _RVMEncoder(backbone_name=backbone_name)

    def build_trainable_model(self, device: str = "cpu"):
        """Build an unfrozen PyTorch model for training."""
        if torch is None:
            raise ImportError("PyTorch is required to build a trainable model.")

        model = self._build_rvm_model(self.backbone)
        model.to(self._resolve_device(str(device)))
        return model

    @staticmethod
    def _strip_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        for key, value in state_dict.items():
            cleaned[key.replace("module.", "").replace("model.", "")] = value
        return cleaned

    def load_rvm_backbone(self, weights_path: str, device: str = "cuda"):
        """Load a Ready-to-infer RVM model or ONNX session.

        Args:
            weights_path: Path to a .pth or .onnx weight file.
            device: Target device name such as 'cuda' or 'cpu'.

        Returns:
            torch.nn.Module for .pth checkpoints or onnxruntime.InferenceSession for .onnx.
        """
        if not weights_path or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        weights_path = os.fspath(weights_path)
        lower_name = weights_path.lower()
        resolved_device = self._resolve_device(device)

        if lower_name.endswith(".onnx"):
            if ort is None:
                raise ImportError("onnxruntime is required to load ONNX weights.")

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if resolved_device == "cuda" else ["CPUExecutionProvider"]
            session = ort.InferenceSession(weights_path, providers=providers)
            return session

        if torch is None:
            raise ImportError("PyTorch is required to load .pth weights.")

        model = self._build_rvm_model(self.backbone)
        checkpoint = torch.load(weights_path, map_location=resolved_device)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = self._strip_prefix(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict):
            state_dict = self._strip_prefix(checkpoint)
        elif hasattr(checkpoint, "state_dict"):
            state_dict = self._strip_prefix(checkpoint.state_dict())
        else:
            raise TypeError("Unsupported checkpoint format. Expected a state_dict or model checkpoint.")

        model.load_state_dict(state_dict, strict=False)
        model.to(resolved_device)

        if self.half_precision and resolved_device == "cuda":
            model = model.half()

        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        torch.set_grad_enabled(False)
        return model


__all__ = ["MattingModelBuilder"]
