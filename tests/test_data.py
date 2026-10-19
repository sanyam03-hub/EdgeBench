from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from conftest import IMAGES_PER_CLASS, WNIDS
from PIL import Image

from edgebench.config import SubsetConfig
from edgebench.data import (
    DOWNLOAD_COMMAND,
    IMAGENETTE_WNID_TO_IMAGENET_IDX,
    ArrayCalibrationReader,
    ImagenetteCalibrationReader,
    calibration_subset,
    eval_subset,
    label_of,
    list_images,
    load_batch,
    locate_imagenette,
    preprocess,
    sample_subset,
    synthetic_split,
)
from edgebench.errors import DatasetError

REAL_ROOT = Path(__file__).resolve().parents[1] / "data" / "imagenette2-160"


def fake_transforms(image: Image.Image) -> np.ndarray:
    """Numpy stand-in for torchvision transforms: resize to 16x16, CHW, scale to [0, 1]."""
    return np.asarray(image.resize((16, 16)), dtype=np.float32).transpose(2, 0, 1) / 255.0


def test_mapping_matches_imagenet_indices() -> None:
    assert IMAGENETTE_WNID_TO_IMAGENET_IDX == {
        "n01440764": 0,
        "n02102040": 217,
        "n02979186": 482,
        "n03000684": 491,
        "n03028079": 497,
        "n03394916": 566,
        "n03417042": 569,
        "n03425413": 571,
        "n03445777": 574,
        "n03888257": 701,
    }


def test_locate_returns_root(synthetic_imagenette: Path) -> None:
    assert locate_imagenette(synthetic_imagenette.parent) == synthetic_imagenette


def test_locate_uses_env_var(synthetic_imagenette: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDGEBENCH_DATA_DIR", str(synthetic_imagenette.parent))
    assert locate_imagenette() == synthetic_imagenette


def test_missing_dataset_error_has_download_command(tmp_path: Path) -> None:
    with pytest.raises(DatasetError) as exc_info:
        locate_imagenette(tmp_path)
    assert DOWNLOAD_COMMAND in str(exc_info.value)
    assert "imagenette2-160.tgz" in DOWNLOAD_COMMAND


def test_list_images_sorted_and_images_only(synthetic_imagenette: Path) -> None:
    paths = list_images(synthetic_imagenette / "val")
    assert len(paths) == len(WNIDS) * IMAGES_PER_CLASS
    assert paths == sorted(paths)
    assert all(p.suffix == ".JPEG" for p in paths)


def test_list_images_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetError):
        list_images(tmp_path / "nope")


def test_sample_subset_deterministic(synthetic_imagenette: Path) -> None:
    paths = list_images(synthetic_imagenette / "val")
    first = sample_subset(paths, 10, seed=0)
    assert first == sample_subset(paths, 10, seed=0)
    assert first != sample_subset(paths, 10, seed=1)
    assert len(set(first)) == 10
    assert first == sorted(first)


def test_sample_subset_too_large_raises(synthetic_imagenette: Path) -> None:
    paths = list_images(synthetic_imagenette / "val")
    with pytest.raises(DatasetError, match="31"):
        sample_subset(paths, 31, seed=0)


def test_calibration_and_eval_subsets_are_disjoint(synthetic_imagenette: Path) -> None:
    cfg = SubsetConfig(n_images=20, batch_size=4, seed=0)
    calib = calibration_subset(synthetic_imagenette, cfg)
    evaluation = eval_subset(synthetic_imagenette, cfg)
    assert all(p.parent.parent.name == "train" for p in calib)
    assert all(p.parent.parent.name == "val" for p in evaluation)
    assert not set(calib) & set(evaluation)


def test_label_of_maps_every_class(synthetic_imagenette: Path) -> None:
    for path in list_images(synthetic_imagenette / "train"):
        assert label_of(path) == IMAGENETTE_WNID_TO_IMAGENET_IDX[path.parent.name]


def test_label_of_unknown_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="n99999999"):
        label_of(tmp_path / "n99999999" / "x.JPEG")


def test_preprocess_shape_and_dtype(synthetic_imagenette: Path) -> None:
    arr = preprocess(list_images(synthetic_imagenette / "val")[0], fake_transforms)
    assert arr.dtype == np.float32
    assert arr.shape == (3, 16, 16)


def test_preprocess_rejects_bad_transform_output(synthetic_imagenette: Path) -> None:
    path = list_images(synthetic_imagenette / "val")[0]
    with pytest.raises(DatasetError, match=r"\(3, H, W\)"):
        preprocess(path, lambda image: np.zeros((16, 16), dtype=np.float32))


def test_load_batch(synthetic_imagenette: Path) -> None:
    paths = list_images(synthetic_imagenette / "val")[:5]
    batch, labels = load_batch(paths, fake_transforms)
    assert batch.shape == (5, 3, 16, 16)
    assert batch.dtype == np.float32
    assert labels.dtype == np.int64
    assert labels.tolist() == [label_of(p) for p in paths]


def test_calibration_reader_yields_batches_then_none(synthetic_imagenette: Path) -> None:
    paths = list_images(synthetic_imagenette / "train")[:10]
    reader = ImagenetteCalibrationReader("input", paths, fake_transforms, batch_size=4)
    assert len(reader) == 3
    sizes = []
    while (item := reader.get_next()) is not None:
        assert list(item) == ["input"]
        sizes.append(item["input"].shape[0])
    assert sizes == [4, 4, 2]
    assert reader.get_next() is None


def test_calibration_reader_is_an_ort_reader(synthetic_imagenette: Path) -> None:
    calibrate = pytest.importorskip("onnxruntime.quantization")
    reader = ImagenetteCalibrationReader("input", [], fake_transforms, batch_size=4)
    assert isinstance(reader, calibrate.CalibrationDataReader)


def test_synthetic_split_deterministic_and_labelled() -> None:
    x, labels = synthetic_split(12, (3, 8, 8), seed=0, split="val")
    x2, labels2 = synthetic_split(12, (3, 8, 8), seed=0, split="val")
    assert x.shape == (12, 3, 8, 8) and x.dtype == np.float32
    assert labels.dtype == np.int64
    assert np.array_equal(x, x2) and np.array_equal(labels, labels2)
    assert set(labels.tolist()) == set(IMAGENETTE_WNID_TO_IMAGENET_IDX.values())


def test_synthetic_train_and_val_differ_for_same_seed() -> None:
    train, _ = synthetic_split(4, (3, 8, 8), seed=0, split="train")
    val, _ = synthetic_split(4, (3, 8, 8), seed=0, split="val")
    assert not np.array_equal(train, val)


def test_synthetic_split_rejects_unknown_split() -> None:
    with pytest.raises(DatasetError, match="test"):
        synthetic_split(4, (3, 8, 8), seed=0, split="test")


def test_array_calibration_reader() -> None:
    arr = np.zeros((10, 3, 8, 8), dtype=np.float32)
    reader = ArrayCalibrationReader("input", arr, batch_size=4)
    assert len(reader) == 3
    sizes = []
    while (item := reader.get_next()) is not None:
        sizes.append(item["input"].shape[0])
    assert sizes == [4, 4, 2]


@pytest.mark.data
@pytest.mark.skipif(not REAL_ROOT.is_dir(), reason="Imagenette not downloaded to data/")
def test_real_dataset_val_has_10_classes_and_1000_subset_loads() -> None:
    from torchvision.models import MobileNet_V3_Small_Weights

    root = locate_imagenette(REAL_ROOT.parent)
    assert sorted(p.name for p in (root / "val").iterdir()) == sorted(WNIDS)
    cfg = SubsetConfig(n_images=1000, batch_size=32, seed=0)
    paths = eval_subset(root, cfg)
    transforms = MobileNet_V3_Small_Weights.DEFAULT.transforms()
    # Loaded per batch (as evaluation does): the whole subset at once needs >1 GB of RAM.
    counts: Counter[int] = Counter()
    for start in range(0, len(paths), cfg.batch_size):
        batch, labels = load_batch(paths[start : start + cfg.batch_size], transforms)
        assert batch.shape[1:] == (3, 224, 224)
        counts.update(labels.tolist())
    assert sum(counts.values()) == 1000
    assert len(counts) == 10
    print(f"\nlabel distribution (ImageNet idx: count): {dict(sorted(counts.items()))}")
