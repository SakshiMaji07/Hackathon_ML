from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.segmentation.rvm_model import MattingModelBuilder
from training.dataset import MattingDataset
from training.loss import MattingLoss


def _seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class MattingTrainer:
    """Train and validate the project's alpha-prediction matting model."""

    def __init__(self, config_path):
        self.config_path = Path(config_path).expanduser().resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Training configuration not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        if not isinstance(config, dict):
            raise ValueError("The YAML configuration must contain a mapping at its root.")
        self.config: dict[str, Any] = config
        self.project_dir = self.config_path.parent.parent
        training = self._section("training")
        data = self._section("data")
        model_config = self._section("model")

        self.seed = int(training.get("seed", config.get("seed", 0)))
        self._seed_everything(self.seed)
        requested_device = str(training.get("device", config.get("device", "cuda"))).lower()
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        elif requested_device != "cpu" and not requested_device.startswith("cuda"):
            raise ValueError("training.device must be 'cpu' or a CUDA device such as 'cuda:0'.")
        self.device = torch.device(requested_device)

        train_dataset = self._make_dataset(data, "train")
        val_dataset = self._make_dataset(data, "val")

        batch_size = int(training.get("batch_size", 1))
        num_workers = int(training.get("num_workers", 0))
        if batch_size < 1 or num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers cannot be negative.")

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        loader_options = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": self.device.type == "cuda",
            "worker_init_fn": _seed_worker,
        }
        self.train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            generator=generator,
            **loader_options,
        )
        self.val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_options,
        )

        self.model = MattingModelBuilder(
            backbone=str(model_config.get("backbone", "mobilenet_v3_small"))
        ).build_trainable_model(device=str(self.device))
        pretrained_path = model_config.get("pretrained_checkpoint")
        if pretrained_path:
            checkpoint_path = self._resolve_path(pretrained_path)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
            self.model.load_state_dict(state_dict, strict=False)
        self.model.train()

        learning_rate = float(training.get("lr", 1e-4))
        if learning_rate <= 0:
            raise ValueError("training.lr must be greater than zero.")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = self._make_scheduler(training)
        self.scheduler_per_step = bool(
            self._section("scheduler").get("per_step", False)
        )
        self.log_interval = int(training.get("log_interval", 10))
        if self.log_interval < 1:
            raise ValueError("training.log_interval must be positive.")

        loss_config = self._section("loss")
        self.criterion = MattingLoss(
            w1=float(loss_config.get("w1", 1.0)),
            w2=float(loss_config.get("w2", 1.0)),
            w3=float(loss_config.get("w3", 1.0)),
        ).to(self.device)
        self.gradient_clip_norm = training.get("gradient_clip_norm")
        if self.gradient_clip_norm is not None:
            self.gradient_clip_norm = float(self.gradient_clip_norm)
            if self.gradient_clip_norm <= 0:
                raise ValueError("training.gradient_clip_norm must be greater than zero.")

        checkpoint_dir = config.get(
            "checkpoint_dir",
            training.get("checkpoint_dir", "models/weights"),
        )
        self.checkpoint_dir = self._resolve_path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.writer = None
        self.wandb_run = None
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.start_epoch = 0

        resume_path = training.get("resume_from_checkpoint", config.get("resume_from_checkpoint"))
        if resume_path:
            self._resume(self._resolve_path(resume_path))
        self._init_logger()

    def _section(self, name: str) -> dict[str, Any]:
        section = self.config.get(name, {})
        if not isinstance(section, dict):
            raise ValueError(f"Configuration section '{name}' must be a mapping.")
        return section

    def _resolve_path(self, path: str | Path) -> Path:
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = self.project_dir / resolved
        return resolved.resolve()

    def _make_dataset(self, data_config: dict[str, Any], split: str) -> MattingDataset:
        split_config = data_config.get(split)
        if not isinstance(split_config, dict):
            raise ValueError(
                f"Configuration must define data.{split} with fg_dir, alpha_dir, and bg_dir."
            )
        missing = [key for key in ("fg_dir", "alpha_dir", "bg_dir") if key not in split_config]
        if missing:
            raise ValueError(f"Missing data.{split} path(s): {', '.join(missing)}.")

        paths = [self._resolve_path(split_config[key]) for key in ("fg_dir", "alpha_dir", "bg_dir")]
        return MattingDataset(*paths)

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _make_scheduler(self, training: dict[str, Any]):
        scheduler_config = self._section("scheduler")
        name = str(scheduler_config.get("name", "cosine")).lower()
        epochs = int(training.get("epochs", 1))
        if epochs < 1:
            raise ValueError("training.epochs must be positive.")
        if name in {"none", "off"}:
            return None
        if name == "cosine":
            t_max = int(scheduler_config.get("t_max", epochs))
            if t_max < 1:
                raise ValueError("scheduler.t_max must be positive.")
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=float(scheduler_config.get("eta_min", 0.0)),
            )
        if name == "step":
            step_size = int(scheduler_config.get("step_size", max(1, epochs // 3)))
            if step_size < 1:
                raise ValueError("scheduler.step_size must be positive.")
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=float(scheduler_config.get("gamma", 0.1)),
            )
        raise ValueError(f"Unsupported scheduler '{name}'. Use 'cosine', 'step', or 'none'.")

    def _init_logger(self) -> None:
        logging_config = self._section("logging")
        backend = str(logging_config.get("backend", "tensorboard")).lower()
        if backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            log_dir = self._resolve_path(logging_config.get("log_dir", "runs/matting"))
            self.writer = SummaryWriter(log_dir=str(log_dir))
        elif backend == "wandb":
            try:
                import wandb
            except ImportError as exc:
                raise ImportError(
                    "Install wandb to use logging.backend: wandb."
                ) from exc

            self.wandb_run = wandb.init(
                project=logging_config.get("project", "matting"),
                name=logging_config.get("run_name"),
                config=self.config,
            )
        elif backend != "none":
            raise ValueError(
                f"Unsupported logging backend '{backend}'. "
                "Use 'tensorboard', 'wandb', or 'none'."
            )

    def _log_scalars(self, values: dict[str, float], step: int) -> None:
        if self.writer is not None:
            for name, value in values.items():
                self.writer.add_scalar(name, value, step)
        if self.wandb_run is not None:
            self.wandb_run.log(values, step=step)

    @staticmethod
    def _extract_alpha(model_output: Any) -> torch.Tensor:
        if isinstance(model_output, dict):
            for key in ("alpha", "pred_alpha", "alpha_map"):
                if key in model_output:
                    return model_output[key]
            raise ValueError("Model output mapping does not contain an alpha tensor.")
        if isinstance(model_output, (tuple, list)):
            if not model_output:
                raise ValueError("Model returned an empty output sequence.")
            model_output = model_output[0]
        if not isinstance(model_output, torch.Tensor):
            raise TypeError(
                f"Expected model output to contain a tensor alpha prediction, "
                f"got {type(model_output).__name__}."
            )
        return model_output

    def _predict_alpha(self, composite: torch.Tensor) -> torch.Tensor:
        logits = self._extract_alpha(self.model(composite))
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(
                f"Expected model alpha logits with shape (B, 1, H, W), got {tuple(logits.shape)}."
            )
        if logits.shape[-2:] != composite.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=composite.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return torch.sigmoid(logits)

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        self.train_loader.dataset.seed = self.seed + epoch
        total_loss = 0.0
        sample_count = 0

        for batch in self.train_loader:
            composite = batch["composite"].to(self.device, non_blocking=True)
            alpha_target = batch["alpha"].to(self.device, non_blocking=True)
            foreground = batch["fg"].to(self.device, non_blocking=True)
            background = batch["bg"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():
                pred_alpha = self._predict_alpha(composite)
                # This model predicts alpha only; the dataset foreground is used
                # as the known foreground for the composite-fidelity term.
                loss = self.criterion(
                    {"alpha": pred_alpha, "fg": foreground},
                    {
                        "alpha": alpha_target,
                        "bg": background,
                        "composite": composite,
                    },
                )
                loss.backward()
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_norm,
                )
            self.optimizer.step()

            batch_size = composite.shape[0]
            total_loss += loss.detach().item() * batch_size
            sample_count += batch_size
            self.global_step += 1
            if self.scheduler is not None and self.scheduler_per_step:
                self.scheduler.step()
            if self.global_step % self.log_interval == 0:
                self._log_scalars(
                    {"train/running_loss": total_loss / sample_count},
                    self.global_step,
                )

        if sample_count == 0:
            raise ValueError("Training DataLoader produced no batches.")
        average_loss = total_loss / sample_count
        self._log_scalars({"train/loss": average_loss}, self.global_step)
        return average_loss

    def validate(self, epoch: int) -> tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        total_absolute_error = 0.0
        total_squared_error = 0.0
        sample_count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                composite = batch["composite"].to(self.device, non_blocking=True)
                alpha_target = batch["alpha"].to(self.device, non_blocking=True)
                foreground = batch["fg"].to(self.device, non_blocking=True)
                background = batch["bg"].to(self.device, non_blocking=True)
                pred_alpha = self._predict_alpha(composite)
                loss = self.criterion(
                    {"alpha": pred_alpha, "fg": foreground},
                    {
                        "alpha": alpha_target,
                        "bg": background,
                        "composite": composite,
                    },
                )

                batch_size = composite.shape[0]
                difference = pred_alpha - alpha_target
                total_loss += loss.item() * batch_size
                total_absolute_error += difference.abs().mean().item() * batch_size
                total_squared_error += difference.square().mean().item() * batch_size
                sample_count += batch_size

        if sample_count == 0:
            raise ValueError("Validation DataLoader produced no batches.")
        metrics = (
            total_loss / sample_count,
            total_absolute_error / sample_count,
            total_squared_error / sample_count,
        )
        self._log_scalars(
            {
                "epoch": float(epoch),
                "val/loss": metrics[0],
                "val/mae": metrics[1],
                "val/mse": metrics[2],
            },
            self.global_step,
        )
        return metrics

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": epoch,
            "global_step": self.global_step,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        self._save_checkpoint_file(checkpoint, self.checkpoint_dir / "latest_checkpoint.pth")
        if is_best:
            self._save_checkpoint_file(checkpoint, self.checkpoint_dir / "best_model.pth")

    def _save_checkpoint_file(self, checkpoint: dict[str, Any], path: Path) -> None:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)

    def _resume(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(
                f"Unsupported resume checkpoint: {checkpoint_path}. "
                "Expected a mapping containing 'model_state_dict'."
            )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)
        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        self.best_val_loss = float(checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf"))))
        self.global_step = int(checkpoint.get("global_step", 0))

    def fit(self) -> None:
        epochs = int(self._section("training").get("epochs", 1))
        try:
            for epoch in range(self.start_epoch, epochs):
                train_loss = self.train_one_epoch(epoch)
                val_loss, val_mae, val_mse = self.validate(epoch)
                if self.scheduler is not None and not self.scheduler_per_step:
                    self.scheduler.step()
                self.save_checkpoint(epoch, val_loss)
                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                    f"val_mae={val_mae:.6f} val_mse={val_mse:.6f}"
                )
                self._log_scalars(
                    {"train/learning_rate": self.optimizer.param_groups[0]["lr"]},
                    self.global_step,
                )
        finally:
            if self.writer is not None:
                self.writer.close()
            if self.wandb_run is not None:
                self.wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the matting model.")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to the training YAML configuration.",
    )
    args = parser.parse_args()
    MattingTrainer(args.config).fit()


if __name__ == "__main__":
    main()