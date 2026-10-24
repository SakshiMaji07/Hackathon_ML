import numpy as np
import pytest

from src.compositing.alpha_blender import CanvasComposer


@pytest.fixture
def composer() -> CanvasComposer:
    height, width = 8, 10
    instance = CanvasComposer.__new__(CanvasComposer)
    instance.target_resolution = (width, height)
    instance.bg = np.empty((height, width, 3), dtype=np.float32)
    instance.bg[:] = (255, 0, 0)  # Blue in BGR order.
    return instance


@pytest.fixture
def solid_images() -> tuple[np.ndarray, np.ndarray]:
    height, width = 8, 10
    foreground = np.empty((height, width, 3), dtype=np.uint8)
    foreground[:] = (0, 0, 255)  # Red in BGR order.
    other_foreground = np.empty((height, width, 3), dtype=np.uint8)
    other_foreground[:] = (0, 255, 255)  # Yellow in BGR order.
    return foreground, other_foreground


class TestCanvasComposer:
    def test_blending_math(
        self,
        composer: CanvasComposer,
        solid_images: tuple[np.ndarray, np.ndarray],
    ) -> None:
        foreground, other_foreground = solid_images
        height, width = foreground.shape[:2]
        opaque = np.ones((height, width), dtype=np.float32)
        transparent = np.zeros((height, width), dtype=np.float32)

        fully_foreground = composer.composite_dual_subjects(
            foreground,
            opaque,
            other_foreground,
            transparent,
            z_order="A_over_B",
        )
        fully_background = composer.composite_dual_subjects(
            foreground,
            transparent,
            other_foreground,
            transparent,
            z_order="A_over_B",
        )

        assert np.array_equal(fully_foreground, foreground)
        assert np.array_equal(
            fully_background,
            np.full_like(foreground, (255, 0, 0)),
        )

    def test_z_order_reversal(
        self,
        composer: CanvasComposer,
        solid_images: tuple[np.ndarray, np.ndarray],
    ) -> None:
        foreground_a, foreground_b = solid_images
        height, width = foreground_a.shape[:2]
        alpha_a = np.zeros((height, width), dtype=np.float32)
        alpha_b = np.zeros((height, width), dtype=np.float32)
        alpha_a[:, : width // 2] = 1.0
        alpha_b[:, : width // 2] = 1.0

        a_over_b = composer.composite_dual_subjects(
            foreground_a,
            alpha_a,
            foreground_b,
            alpha_b,
            z_order="A_over_B",
        )
        b_over_a = composer.composite_dual_subjects(
            foreground_a,
            alpha_a,
            foreground_b,
            alpha_b,
            z_order="B_over_A",
        )
        overlap = (slice(None), slice(0, width // 2))

        assert np.all(a_over_b[overlap] == (0, 0, 255))
        assert np.all(b_over_a[overlap] == (0, 255, 255))
        assert not np.array_equal(a_over_b[overlap], b_over_a[overlap])

    def test_rejects_foreground_size_mismatch(
        self,
        composer: CanvasComposer,
        solid_images: tuple[np.ndarray, np.ndarray],
    ) -> None:
        foreground, other_foreground = solid_images
        wrong_size_foreground = foreground[:-1]
        alpha = np.ones(foreground.shape[:2], dtype=np.float32)

        with pytest.raises(ValueError, match="does not match target"):
            composer.composite_dual_subjects(
                wrong_size_foreground,
                alpha,
                other_foreground,
                alpha,
                z_order="A_over_B",
            )