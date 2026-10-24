from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class MattingDataset(Dataset):
    """Generate training examples by compositing foregrounds over backgrounds."""

    def __init__(self, fg_dir, alpha_dir, bg_dir, transform=None):
        self.fg_dir = Path(fg_dir)
        self.alpha_dir = Path(alpha_dir)
        self.bg_dir = Path(bg_dir)
        self.seed = 0

        for directory in (self.fg_dir, self.alpha_dir, self.bg_dir):
            if not directory.is_dir():
                raise FileNotFoundError(f"Image directory does not exist: {directory}")

        foregrounds = self._images_by_stem(self.fg_dir)
        alphas = self._images_by_stem(self.alpha_dir)
        matched_stems = sorted(foregrounds.keys() & alphas.keys())
        missing_alpha = sorted(foregrounds.keys() - alphas.keys())
        missing_foreground = sorted(alphas.keys() - foregrounds.keys())

        if missing_alpha or missing_foreground:
            warnings.warn(
                "Skipping unmatched foreground/alpha files "
                f"({len(missing_alpha)} foregrounds without alpha, "
                f"{len(missing_foreground)} alpha masks without foreground).",
                RuntimeWarning,
                stacklevel=2,
            )

        self.fg_files = [foregrounds[stem] for stem in matched_stems]
        self.alpha_files = [alphas[stem] for stem in matched_stems]
        if not self.fg_files:
            raise ValueError(
                f"No matching foreground/alpha filename stems in "
                f"{self.fg_dir} and {self.alpha_dir}."
            )

        self.bg_files = sorted(
            path
            for path in self.bg_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not self.bg_files:
            raise ValueError(f"No background images found in {self.bg_dir}.")

        self.transform = transform if transform is not None else A.Compose(
            [
                A.Affine(
                    scale=(0.8, 1.2),
                    translate_percent=(-0.1, 0.1),
                    rotate=(-15, 15),
                    p=0.7,
                ),
                A.ColorJitter(p=0.5),
                A.HorizontalFlip(p=0.5),
            ]
        )

    @staticmethod
    def _images_by_stem(directory: Path) -> dict[str, Path]:
        images: dict[str, Path] = {}
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            stem = path.stem.casefold()
            if stem in images:
                raise ValueError(
                    f"Multiple images with the same filename stem '{path.stem}' "
                    f"found in {directory}."
                )
            images[stem] = path
        return images

    def __len__(self):
        return len(self.fg_files)

    def _sample_rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self.seed, idx]))

    @staticmethod
    def _read_image(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _read_alpha(path: Path) -> np.ndarray:
        alpha = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if alpha is None:
            raise OSError(f"Could not read alpha mask: {path}")

        if alpha.ndim == 3:
            if alpha.shape[2] == 4:
                alpha = cv2.cvtColor(alpha, cv2.COLOR_BGRA2GRAY)
            elif alpha.shape[2] == 3:
                alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
            else:
                raise ValueError(f"Unsupported alpha image shape {alpha.shape}: {path}")
        elif alpha.ndim != 2:
            raise ValueError(f"Unsupported alpha image shape {alpha.shape}: {path}")

        if np.issubdtype(alpha.dtype, np.integer):
            alpha = alpha.astype(np.float32) / np.iinfo(alpha.dtype).max
        else:
            alpha = alpha.astype(np.float32)
            if alpha.size and alpha.max() > 1.0:
                alpha /= 255.0

        alpha = np.clip(alpha, 0.0, 1.0)
        return np.rint(alpha * 255.0).astype(np.uint8)

    @staticmethod
    def _read_background(path: Path, height: int, width: int, rng: np.random.Generator) -> np.ndarray:
        background = MattingDataset._read_image(path)
        bg_height, bg_width = background.shape[:2]

        # Scale to cover the foreground dimensions, then crop the excess.
        scale = max(height / bg_height, width / bg_width)
        resized_height = max(height, int(np.ceil(bg_height * scale)))
        resized_width = max(width, int(np.ceil(bg_width * scale)))
        background = cv2.resize(
            background,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

        top = int(rng.integers(0, resized_height - height + 1))
        left = int(rng.integers(0, resized_width - width + 1))
        return background[top : top + height, left : left + width]

    def _augment(self, foreground: np.ndarray, alpha: np.ndarray, idx: int) -> tuple[np.ndarray, np.ndarray]:
        seed = int(np.random.SeedSequence([self.seed, idx]).generate_state(1)[0])
        set_seed = getattr(self.transform, "set_random_seed", None)

        if callable(set_seed):
            set_seed(seed)
            augmented: dict[str, Any] = self.transform(image=foreground, mask=alpha)
        else:
            python_state = random.getstate()
            numpy_state = np.random.get_state()
            random.seed(seed)
            np.random.seed(seed)
            try:
                augmented = self.transform(image=foreground, mask=alpha)
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)

        if not isinstance(augmented, dict) or "image" not in augmented or "mask" not in augmented:
            raise ValueError("Transform must return a mapping containing 'image' and 'mask'.")

        augmented_foreground = np.asarray(augmented["image"])
        augmented_alpha = np.asarray(augmented["mask"])
        if (
            augmented_foreground.ndim != 3
            or augmented_foreground.shape[2] != 3
            or augmented_alpha.ndim != 2
            or augmented_foreground.shape[:2] != augmented_alpha.shape
        ):
            raise ValueError(
                "Transform must preserve aligned HxWx3 image and HxW mask arrays; "
                f"got {augmented_foreground.shape} and {augmented_alpha.shape}."
            )

        return augmented_foreground, augmented_alpha

    @staticmethod
    def _to_unit_float(array: np.ndarray) -> np.ndarray:
        array = array.astype(np.float32)
        if array.size and array.max() > 1.0:
            array /= 255.0
        return np.clip(array, 0.0, 1.0)

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Dataset index out of range: {idx}")

        foreground = self._read_image(self.fg_files[idx])
        alpha = self._read_alpha(self.alpha_files[idx])
        if foreground.shape[:2] != alpha.shape:
            raise ValueError(
                f"Foreground and alpha dimensions do not match for "
                f"{self.fg_files[idx].name}: {foreground.shape[:2]} vs {alpha.shape}."
            )

        rng = self._sample_rng(idx)
        background_path = self.bg_files[int(rng.integers(len(self.bg_files)))]
        background = self._read_background(
            background_path,
            foreground.shape[0],
            foreground.shape[1],
            rng,
        )
        foreground, alpha = self._augment(foreground, alpha, idx)

        foreground = self._to_unit_float(foreground)
        alpha = self._to_unit_float(alpha)[..., None]
        background = self._to_unit_float(background)
        composite = alpha * foreground + (1.0 - alpha) * background

        def to_tensor(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float()

        return {
            "composite": to_tensor(composite),
            "alpha": to_tensor(alpha),
            "fg": to_tensor(foreground),
            "bg": to_tensor(background),
        }