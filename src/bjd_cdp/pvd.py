"""Deterministic implementations of the released PVD operators.

Images are OpenCV BGR uint8 arrays. Pass a fixed NumPy Generator to reproduce
the stochastic noise draws for a given input order.
"""

from __future__ import annotations

import cv2
import numpy as np


def _clip(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def _gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return _clip(image.astype(np.float32) + rng.normal(0, sigma, image.shape))


def _salt_pepper(
    image: np.ndarray,
    amount: float,
    rng: np.random.Generator,
    salt_fraction: float = 0.5,
) -> np.ndarray:
    result = image.copy()
    height, width = result.shape[:2]
    count = int(amount * height * width)
    salt_count = int(count * salt_fraction)
    pepper_count = count - salt_count
    salt_y = rng.integers(0, height, salt_count)
    salt_x = rng.integers(0, width, salt_count)
    result[salt_y, salt_x] = 255
    pepper_y = rng.integers(0, height, pepper_count)
    pepper_x = rng.integers(0, width, pepper_count)
    result[pepper_y, pepper_x] = 0
    return result


def _contrast(image: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return _clip(image.astype(np.float32) * alpha + beta)


def _resize_degrade(image: np.ndarray, scale: float) -> np.ndarray:
    height, width = image.shape[:2]
    small = cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def _motion_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(image, -1, kernel)


def sp_text(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Degrade the top and bottom 26% regions, with a mild global contrast pass."""
    result = image.copy()
    height = result.shape[0]
    region_height = int(height * 0.26)
    for region in (slice(0, region_height), slice(height - region_height, height)):
        patch = _salt_pepper(result[region].copy(), 0.08, rng)
        patch = _gaussian_noise(patch, 28.0, rng)
        patch = cv2.GaussianBlur(patch, (7, 7), 2.5)
        result[region] = _contrast(patch, 0.68, 6.0)
    return _contrast(result, 0.93, 2.0)


def sp_image(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = _gaussian_noise(image.copy(), 34.0, rng)
    result = _salt_pepper(result, 0.07, rng)
    return _contrast(result, 0.82, 4.0)


def spread(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    del rng  # the released spread operator is deterministic
    result = _resize_degrade(image.copy(), 0.50)
    result = cv2.GaussianBlur(result, (15, 15), 5.5)
    result = _motion_blur(result, 15)
    return _contrast(result, 0.78, 5.0)


def newsprint(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    result = _resize_degrade(result, 0.58)
    result = _gaussian_noise(result, 14.0, rng)
    result = _salt_pepper(result, 0.025, rng)
    result = cv2.GaussianBlur(result, (5, 5), 1.2)
    return _contrast(result, 0.72, 10.0)


OPERATORS = {
    "sp_text": sp_text,
    "sp_image": sp_image,
    "spread": spread,
    "newsprint": newsprint,
}

