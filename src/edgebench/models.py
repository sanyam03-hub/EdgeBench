"""Model registry: name -> PyTorch module (eval mode) plus its preprocessing transforms.

Does not export or run inference.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torchvision import models as tv_models
from torchvision import transforms as tv_transforms

from edgebench.errors import ConfigError, DatasetError

TINY_CNN_SEED = 0
TINY_CNN_SIZE = 32


@dataclass(frozen=True)
class LoadedModel:
    """A model ready for inference and the transforms that produce its (3, H, W) input."""

    name: str
    module: nn.Module
    transforms: Callable[[Any], Any]
    num_classes: int
    input_shape: tuple[int, int, int]


class TinyCNN(nn.Module):
    """Small random-init CNN for tests and the CI smoke run (no download, 10 classes)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


_TORCHVISION = {
    "mobilenet_v3_small": (tv_models.mobilenet_v3_small, tv_models.MobileNet_V3_Small_Weights),
    "resnet18": (tv_models.resnet18, tv_models.ResNet18_Weights),
}
MODEL_NAMES = (*_TORCHVISION, "tiny_cnn")


def load_model(name: str, weights: str = "DEFAULT") -> LoadedModel:
    """Build a registered model in eval mode.

    Args:
        name: A key of ``MODEL_NAMES``.
        weights: torchvision weights enum member name (ignored for ``tiny_cnn``).

    Raises:
        ConfigError: Unknown model or weights name.
        DatasetError: Pretrained weights could not be downloaded or read.
    """
    if name == "tiny_cnn":
        return _tiny_cnn()
    if name not in _TORCHVISION:
        raise ConfigError(f"Unknown model {name!r}; registered models: {list(MODEL_NAMES)}")
    constructor, weights_enum = _TORCHVISION[name]
    member = weights_enum.__members__.get(weights)
    if member is None:
        allowed = list(weights_enum.__members__)
        raise ConfigError(f"Unknown weights {weights!r} for {name}; allowed: {allowed}")
    try:
        module = constructor(weights=member)
    except OSError as exc:
        raise DatasetError(
            f"Could not load pretrained weights for {name} ({member}): {exc}. Check the network "
            "connection, or pre-populate the cache directory set by TORCH_HOME."
        ) from exc
    transforms = member.transforms()
    crop = int(transforms.crop_size[0])
    return LoadedModel(
        name=name,
        module=module.eval(),
        transforms=transforms,
        num_classes=len(member.meta["categories"]),
        input_shape=(3, crop, crop),
    )


def _tiny_cnn() -> LoadedModel:
    with torch.random.fork_rng():
        torch.manual_seed(TINY_CNN_SEED)
        module = TinyCNN()
    transforms = tv_transforms.Compose(
        [tv_transforms.Resize((TINY_CNN_SIZE, TINY_CNN_SIZE)), tv_transforms.ToTensor()]
    )
    return LoadedModel(
        name="tiny_cnn",
        module=module.eval(),
        transforms=transforms,
        num_classes=10,
        input_shape=(3, TINY_CNN_SIZE, TINY_CNN_SIZE),
    )
