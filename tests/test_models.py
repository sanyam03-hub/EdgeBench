from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from edgebench.errors import ConfigError
from edgebench.models import MODEL_NAMES, LoadedModel, load_model


def test_registry_names() -> None:
    assert set(MODEL_NAMES) == {"mobilenet_v3_small", "resnet18", "tiny_cnn"}


def test_tiny_cnn_shape_and_eval_mode(tiny_cnn: LoadedModel) -> None:
    assert tiny_cnn.num_classes == 10
    assert tiny_cnn.input_shape == (3, 32, 32)
    assert not tiny_cnn.module.training
    with torch.inference_mode():
        out = tiny_cnn.module(torch.zeros(4, *tiny_cnn.input_shape))
    assert out.shape == (4, 10)


def test_tiny_cnn_is_deterministic_and_leaves_global_rng_alone() -> None:
    torch.manual_seed(123)
    expected_next = torch.rand(1)
    torch.manual_seed(123)
    a = load_model("tiny_cnn")
    b = load_model("tiny_cnn")
    assert torch.rand(1) == expected_next
    for pa, pb in zip(a.module.parameters(), b.module.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_tiny_cnn_transforms_produce_input_shape(tiny_cnn: LoadedModel) -> None:
    image = Image.fromarray(np.zeros((20, 24, 3), dtype=np.uint8))
    arr = np.asarray(tiny_cnn.transforms(image), dtype=np.float32)
    assert arr.shape == tiny_cnn.input_shape


def test_unknown_model_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="vgg16"):
        load_model("vgg16")


def test_unknown_weights_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="NOPE"):
        load_model("resnet18", weights="NOPE")


@pytest.mark.slow
@pytest.mark.parametrize("name", ["mobilenet_v3_small", "resnet18"])
def test_pretrained_models_load(name: str) -> None:
    loaded = load_model(name)
    assert loaded.num_classes == 1000
    assert loaded.input_shape == (3, 224, 224)
    with torch.inference_mode():
        assert loaded.module(torch.zeros(1, 3, 224, 224)).shape == (1, 1000)
