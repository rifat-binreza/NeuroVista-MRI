"""Notebook-grounded preprocessing. No downloads, datasets or patient images included.

RGB conversion, 256/299 pixel sizes and /255 follow supplied inference code.
Nearest-neighbour interpolation follows Keras load_img's default; it is not a
confirmed training choice. Binary-mask handling is a new explicit convention.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps


def prepare_image(path: str | Path, task: str) -> np.ndarray:
    """Return one float32 NHWC image, normalized to [0,1]."""
    if task not in {"segmentation", "classification"}:
        raise ValueError("task must be segmentation or classification")
    size = 256 if task == "segmentation" else 299
    with Image.open(path) as image:
        if image.width * image.height > 16_000_000:
            raise ValueError("Image exceeds 16 megapixels")
        image = ImageOps.exif_transpose(image).convert("RGB")
        array = np.asarray(image.resize((size, size), Image.Resampling.NEAREST), dtype=np.float32)
    return array[None, ...] / 255.0


def prepare_binary_mask(path: str | Path, threshold: int = 128) -> np.ndarray:
    """Return a 256-square binary mask; only for explicitly binary annotations.

    Do not use on categorical masks whose integer values encode class IDs.
    """
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must lie in [0,255]")
    with Image.open(path) as image:
        array = np.asarray(image.convert("L").resize((256, 256), Image.Resampling.NEAREST))
    return (array >= threshold).astype(np.float32)[None, ..., None]
