from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.segmentation.inference import MattingEngine
from src.segmentation.rvm_model import MattingModelBuilder


class _SmallAlphaModel(torch.nn.Module):
    """Fast deterministic stand-in that emits low-resolution alpha logits."""

    def forward(self, image: torch.Tensor, *_states: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(image[:, :1], kernel_size=8, stride=8)


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> MattingEngine:
    checkpoint_path = tmp_path / "dummy_model.pth"
    checkpoint_path.touch()
    model = _SmallAlphaModel().eval()

    def load_test_model(
        _builder: MattingModelBuilder,
        weights_path: str,
        device: str = "cuda",
    ) -> torch.nn.Module:
        assert Path(weights_path).is_file()
        assert device == "cpu"
        return model

    monkeypatch.setattr(MattingModelBuilder, "load_rvm_backbone", load_test_model)
    return MattingEngine(str(checkpoint_path), device="cpu")


@pytest.fixture
def dummy_batch() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(2, 3, 720, 1280).sigmoid()


class TestMattingEngine:
    def test_model_output_shape(
        self,
        engine: MattingEngine,
        dummy_batch: torch.Tensor,
    ) -> None:
        alpha_output, _ = engine.predict_alpha_batch(dummy_batch)

        assert alpha_output.shape == (2, 1, 720, 1280)
        # MattingEngine processes both camera feeds together; each feed has the
        # requested single-image output shape.
        assert alpha_output[0:1].shape == (1, 1, 720, 1280)
        assert alpha_output[1:2].shape == (1, 1, 720, 1280)

    def test_alpha_range(
        self,
        engine: MattingEngine,
        dummy_batch: torch.Tensor,
    ) -> None:
        alpha_output, _ = engine.predict_alpha_batch(dummy_batch)

        assert torch.all(alpha_output >= 0.0)
        assert torch.all(alpha_output <= 1.0)
