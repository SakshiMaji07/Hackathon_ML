from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.segmentation.rvm_model import MattingModelBuilder


class ModelExporter:
    """Export the trained alpha-prediction model to ONNX and validate it."""

    def __init__(self, model_path, config_path):
        self.config_path = Path(config_path).expanduser().resolve()
        self.model_path = self._resolve_path(model_path)
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Export configuration not found: {self.config_path}")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        with self.config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        if not isinstance(config, dict):
            raise ValueError("The YAML configuration must contain a mapping at its root.")
        self.config: dict[str, Any] = config
        export_config = self._section("export")
        model_config = self._section("model")
        self.export_config = export_config
        self.project_dir = self.config_path.parent.parent

        height, width = self._input_resolution(export_config)
        if height < 32 or width < 32:
            raise ValueError("Export input height and width must be at least 32 pixels.")
        self.input_height = height
        self.input_width = width
        self.opset_version = int(
            export_config.get("opset_version", export_config.get("opset", 17))
        )
        if self.opset_version < 11:
            raise ValueError("ONNX opset_version must be 11 or newer.")

        requested_device = str(export_config.get("device", config.get("device", "cpu"))).lower()
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        elif requested_device != "cpu" and not requested_device.startswith("cuda"):
            raise ValueError("Export device must be 'cpu' or a CUDA device such as 'cuda:0'.")
        self.device = torch.device(requested_device)

        builder = MattingModelBuilder(
            backbone=str(model_config.get("backbone", "mobilenet_v3_small"))
        )
        self.model = builder.build_trainable_model(device=str(self.device))
        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = self._checkpoint_state_dict(checkpoint)
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(
                "Checkpoint contains unexpected model parameters: "
                + ", ".join(incompatible.unexpected_keys)
            )
        if incompatible.missing_keys:
            raise ValueError(
                "Checkpoint is missing model parameters: "
                + ", ".join(incompatible.missing_keys)
            )
        self.model.eval()

        self.input_name = str(export_config.get("input_name", "input"))
        self.output_name = str(export_config.get("output_name", "alpha"))
        self.tolerance = float(export_config.get("verification_tolerance", 1e-3))
        if self.tolerance <= 0:
            raise ValueError("export.verification_tolerance must be positive.")
        self.last_verification: tuple[bool, float] | None = None

    def _resolve_path(self, path: str | Path) -> Path:
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = self.config_path.parent.parent / resolved
        return resolved.resolve()

    def _section(self, name: str) -> dict[str, Any]:
        section = self.config.get(name, {})
        if not isinstance(section, dict):
            raise ValueError(f"Configuration section '{name}' must be a mapping.")
        return section

    @staticmethod
    def _input_resolution(export_config: dict[str, Any]) -> tuple[int, int]:
        resolution = export_config.get("input_resolution")
        if resolution is not None:
            if isinstance(resolution, dict):
                height, width = resolution.get("height"), resolution.get("width")
            elif isinstance(resolution, (list, tuple)) and len(resolution) == 2:
                height, width = resolution
            else:
                raise ValueError(
                    "export.input_resolution must be [height, width] or a "
                    "{height, width} mapping."
                )
        else:
            height = export_config.get("input_height", export_config.get("height", 512))
            width = export_config.get("input_width", export_config.get("width", 512))
        try:
            return int(height), int(width)
        except (TypeError, ValueError) as exc:
            raise ValueError("Export input height and width must be integers.") from exc

    @staticmethod
    def _checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict"):
                if key in checkpoint:
                    checkpoint = checkpoint[key]
                    break
        if not isinstance(checkpoint, dict):
            if hasattr(checkpoint, "state_dict"):
                checkpoint = checkpoint.state_dict()
            else:
                raise TypeError("Checkpoint must contain a model state dictionary.")
        if not all(isinstance(key, str) for key in checkpoint):
            raise TypeError("Model state dictionary keys must be strings.")

        cleaned: dict[str, torch.Tensor] = {}
        for key, value in checkpoint.items():
            while key.startswith(("module.", "model.")):
                key = key.split(".", maxsplit=1)[1]
            cleaned[key] = value
        return cleaned

    def _dummy_input(self) -> torch.Tensor:
        return torch.randn(
            1,
            3,
            self.input_height,
            self.input_width,
            dtype=torch.float32,
            device=self.device,
        )

    def export(self, output_onnx_path) -> Path:
        try:
            import onnx
        except ImportError as exc:
            raise ImportError("Install onnx to export and check ONNX models.") from exc

        output_path = self._resolve_path(output_onnx_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_input = self._dummy_input()
        dynamic_axes = {
            self.input_name: {0: "batch", 2: "height", 3: "width"},
            self.output_name: {0: "batch", 2: "output_height", 3: "output_width"},
        }
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=[self.input_name],
                output_names=[self.output_name],
                dynamic_axes=dynamic_axes,
            )

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        passed, max_difference = self.verify_outputs(self.model, output_path)
        self.last_verification = (passed, max_difference)
        print(
            f"ONNX verification {'passed' if passed else 'failed'}: "
            f"max_abs_diff={max_difference:.8g} (tolerance={self.tolerance:g})"
        )
        if not passed:
            raise RuntimeError(
                f"ONNX output verification failed (max abs diff {max_difference:.8g}, "
                f"tolerance {self.tolerance:g})."
            )

        if bool(self.export_config.get("fp16", False)):
            fp16_path = self.optimize_fp16(output_path)
            passed, max_difference = self.verify_outputs(self.model, fp16_path)
            self.last_verification = (passed, max_difference)
            print(
                f"FP16 ONNX verification {'passed' if passed else 'failed'}: "
                f"max_abs_diff={max_difference:.8g} (tolerance={self.tolerance:g})"
            )
            if not passed:
                raise RuntimeError(
                    f"FP16 ONNX output verification failed (max abs diff "
                    f"{max_difference:.8g}, tolerance {self.tolerance:g})."
                )

        return output_path

    def optimize_fp16(self, onnx_path) -> Path:
        try:
            import onnx
            from onnxconverter_common import float16
        except ImportError as exc:
            raise ImportError(
                "Install onnx and onnxconverter-common to convert an ONNX model to FP16."
            ) from exc

        source_path = self._resolve_path(onnx_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {source_path}")
        fp16_path = source_path.with_name(f"{source_path.stem}_fp16.onnx")
        model = onnx.load(str(source_path))
        converted = float16.convert_float_to_float16(model, keep_io_types=True)
        onnx.checker.check_model(converted)
        onnx.save(converted, str(fp16_path))
        return fp16_path

    def verify_outputs(self, pth_model, onnx_path) -> tuple[bool, float]:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("Install onnxruntime to verify ONNX model outputs.") from exc

        model_path = self._resolve_path(onnx_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        available = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device.type == "cuda" and "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
        session = ort.InferenceSession(str(model_path), providers=providers)
        dummy_input = self._dummy_input()
        input_array = dummy_input.detach().cpu().numpy()
        with torch.no_grad():
            pytorch_output = pth_model(dummy_input)
        if isinstance(pytorch_output, (tuple, list)):
            if not pytorch_output:
                raise ValueError("PyTorch model returned no outputs.")
            pytorch_output = pytorch_output[0]
        elif isinstance(pytorch_output, dict):
            if self.output_name in pytorch_output:
                pytorch_output = pytorch_output[self.output_name]
            elif "alpha" in pytorch_output:
                pytorch_output = pytorch_output["alpha"]
            else:
                raise ValueError("PyTorch model output mapping does not contain alpha.")
        if not isinstance(pytorch_output, torch.Tensor):
            raise TypeError("PyTorch model must return an output tensor for verification.")

        input_names = {item.name for item in session.get_inputs()}
        if self.input_name not in input_names:
            raise ValueError(
                f"ONNX input '{self.input_name}' not found; model inputs are "
                f"{sorted(input_names)}."
            )
        output_names = [item.name for item in session.get_outputs()]
        if self.output_name not in output_names:
            raise ValueError(
                f"ONNX output '{self.output_name}' not found; model outputs are "
                f"{output_names}."
            )
        onnx_output = session.run(
            [self.output_name],
            {self.input_name: input_array},
        )[0]
        pytorch_array = pytorch_output.detach().float().cpu().numpy()
        if pytorch_array.shape != onnx_output.shape:
            return False, float("inf")
        max_difference = float(np.max(np.abs(pytorch_array - onnx_output)))
        return max_difference < self.tolerance, max_difference


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and validate a matting model as ONNX.")
    parser.add_argument("--model", default="models/weights/best_model.pth")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default="models/weights/matting.onnx")
    args = parser.parse_args()

    exporter = ModelExporter(args.model, args.config)
    output_path = exporter.export(args.output)
    print(f"Exported ONNX model to {output_path}")


if __name__ == "__main__":
    main()