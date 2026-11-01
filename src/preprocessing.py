"""One preprocessing convention for train, predict and evaluation."""
from pathlib import Path
import cv2
import numpy as np
import torch

SIDE = 256


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot decode photograph: {path}')
    return cv2.resize(image, (SIDE, SIDE), interpolation=cv2.INTER_AREA)


def read_mask(path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f'Cannot decode mask: {path}')
    return binary_mask(mask)


def binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2 or mask.dtype != np.uint8 or not mask.size:
        raise ValueError('Mask must be a nonempty uint8 grayscale array')
    resized = cv2.resize(mask, (SIDE, SIDE), interpolation=cv2.INTER_NEAREST)
    return (resized >= 128).astype(np.uint8)


def image_tensor(image: np.ndarray) -> torch.Tensor:
    """BGR uint8 256 square -> RGB float32 CHW in [0,1]."""
    if image.shape != (SIDE, SIDE, 3) or image.dtype != np.uint8:
        raise ValueError('Expected uint8 BGR photograph of shape (256,256,3)')
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float().div_(255)


def read_pair(image_path: str | Path, mask_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Decode a corresponding pair and reject mismatched original dimensions."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise ValueError(f'Cannot decode image/mask pair: {image_path}, {mask_path}')
    if image.shape[:2] != mask.shape:
        raise ValueError(f'Original image/mask dimensions differ: {image_path}, {mask_path}')
    return cv2.resize(image, (SIDE, SIDE), interpolation=cv2.INTER_AREA), binary_mask(mask)
