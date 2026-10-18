"""Imagenette: locate the dataset, build deterministic subsets, shared preprocessing.

Calibration images come only from ``train/`` and evaluation images only from ``val/``,
so the two subsets are disjoint by construction. Does not import torch: torchvision
transforms are passed in by the caller and their output is converted with ``np.asarray``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from edgebench.config import SubsetConfig
from edgebench.errors import DatasetError

DATASET_NAME = "imagenette2-160"
DATASET_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
DOWNLOAD_COMMAND = (
    f"curl -L -o data/{DATASET_NAME}.tgz {DATASET_URL} && tar -xzf data/{DATASET_NAME}.tgz -C data"
)
IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png"})

# Imagenette folder WNID -> index in the 1000-class ImageNet output (Data_Schema.md §5).
IMAGENETTE_WNID_TO_IMAGENET_IDX: dict[str, int] = {
    "n01440764": 0,  # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}

Transforms = Callable[[Image.Image], Any]


def locate_imagenette(data_dir: Path | None = None) -> Path:
    """Return the ``imagenette2-160`` root containing ``train/`` and ``val/``.

    Args:
        data_dir: Folder containing ``imagenette2-160/``. Defaults to ``$EDGEBENCH_DATA_DIR``
            or ``./data``.

    Raises:
        DatasetError: If the dataset is missing; the message includes the download command.
    """
    base = data_dir if data_dir is not None else Path(os.environ.get("EDGEBENCH_DATA_DIR", "data"))
    root = base / DATASET_NAME
    missing = [split for split in ("train", "val") if not (root / split).is_dir()]
    if missing:
        raise DatasetError(
            f"Imagenette not found at {root} (missing {', '.join(missing)}/). "
            f"Download it from the project root with:\n  {DOWNLOAD_COMMAND}\n"
            "or set EDGEBENCH_DATA_DIR to the folder that contains imagenette2-160/."
        )
    return root


def list_images(split_dir: Path | str) -> list[Path]:
    """All image files under ``split_dir/<wnid>/``, sorted by path."""
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        raise DatasetError(f"Dataset split folder not found: {split_dir}")
    return sorted(
        p for p in split_dir.glob("*/*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def sample_subset(paths: Sequence[Path], n: int, seed: int) -> list[Path]:
    """Pick ``n`` distinct paths with a seeded RNG; the result keeps the input (sorted) order."""
    if n > len(paths):
        raise DatasetError(f"Requested {n} images but only {len(paths)} are available")
    indices = np.random.default_rng(seed).choice(len(paths), size=n, replace=False)
    return [paths[i] for i in sorted(indices)]


def calibration_subset(root: Path, cfg: SubsetConfig) -> list[Path]:
    """Calibration images, drawn only from ``train/``."""
    return sample_subset(list_images(root / "train"), cfg.n_images, cfg.seed)


def eval_subset(root: Path, cfg: SubsetConfig) -> list[Path]:
    """Evaluation images, drawn only from ``val/``."""
    return sample_subset(list_images(root / "val"), cfg.n_images, cfg.seed)


def label_of(path: Path) -> int:
    """ImageNet class index for an image, from its WNID folder name."""
    wnid = path.parent.name
    if wnid not in IMAGENETTE_WNID_TO_IMAGENET_IDX:
        raise DatasetError(f"Unknown Imagenette class folder {wnid!r} for image {path}")
    return IMAGENETTE_WNID_TO_IMAGENET_IDX[wnid]


def preprocess(path: Path, transforms: Transforms) -> np.ndarray:
    """Load one image as RGB and apply ``transforms``; returns float32 (3, H, W)."""
    try:
        with Image.open(path) as image:
            arr = np.asarray(transforms(image.convert("RGB")), dtype=np.float32)
    except OSError as exc:
        raise DatasetError(f"Cannot read image {path}: {exc}") from exc
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise DatasetError(f"Transforms must produce a (3, H, W) array, got {arr.shape} for {path}")
    return arr


def load_batch(paths: Sequence[Path], transforms: Transforms) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess ``paths`` into float32 (N, 3, H, W) and int64 ImageNet labels (N,)."""
    if not paths:
        raise DatasetError("Cannot load an empty batch")
    batch = np.stack([preprocess(p, transforms) for p in paths])
    labels = np.array([label_of(p) for p in paths], dtype=np.int64)
    return batch, labels


class ImagenetteCalibrationReader:
    """ONNX Runtime calibration data reader over a fixed list of training images.

    ORT's ``CalibrationDataReader`` only requires ``get_next``; its ``__subclasshook__``
    accepts any class that defines it, so onnxruntime is not imported here.
    """

    def __init__(
        self, input_name: str, paths: Sequence[Path], transforms: Transforms, batch_size: int
    ) -> None:
        self._input_name = input_name
        self._transforms = transforms
        self._batches = [list(paths[i : i + batch_size]) for i in range(0, len(paths), batch_size)]
        self._next = 0

    def __len__(self) -> int:
        return len(self._batches)

    def get_next(self) -> dict[str, np.ndarray] | None:
        """Return ``{input_name: batch}`` for the next batch, or None when exhausted."""
        if self._next >= len(self._batches):
            return None
        batch, _ = load_batch(self._batches[self._next], self._transforms)
        self._next += 1
        return {self._input_name: batch}
