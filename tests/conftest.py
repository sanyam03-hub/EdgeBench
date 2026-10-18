from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

WNIDS = (
    "n01440764",
    "n02102040",
    "n02979186",
    "n03000684",
    "n03028079",
    "n03394916",
    "n03417042",
    "n03425413",
    "n03445777",
    "n03888257",
)
IMAGES_PER_CLASS = 3


@pytest.fixture
def synthetic_imagenette(tmp_path: Path) -> Path:
    """A fake ``imagenette2-160`` root: train/ and val/ with the 10 real WNID folders.

    Each folder holds a few small random RGB JPEGs (fixed seed) plus one non-image file.
    """
    rng = np.random.default_rng(0)
    root = tmp_path / "data" / "imagenette2-160"
    for split in ("train", "val"):
        for wnid in WNIDS:
            folder = root / split / wnid
            folder.mkdir(parents=True)
            for i in range(IMAGES_PER_CLASS):
                pixels = rng.integers(0, 256, size=(20, 24, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{split}_{wnid}_{i}.JPEG")
            (folder / "notes.txt").write_text("not an image", encoding="utf-8")
    return root
