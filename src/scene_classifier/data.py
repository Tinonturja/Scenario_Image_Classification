"""Dataset and dataloader construction for the scene classification task.

The dataset is expected in the standard ``torchvision.datasets.ImageFolder``
layout, i.e.::

    seg_train/
        buildings/
        forest/
        glacier/
        mountain/
        sea/
        street/
    seg_test/
        buildings/
        ...

This is the layout used by the Intel Image Classification dataset
(6 scene classes, ~14k train / ~3k test images).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as T


@dataclass
class DataBundle:
    """Container for everything a training run needs from the data."""

    train_dataloader: DataLoader
    test_dataloader: DataLoader
    class_names: List[str]
    class_to_idx: dict


def build_transforms(weights: torchvision.models.WeightsEnum | None = None) -> Callable:
    """Build the image transform pipeline.

    If ``weights`` is given (e.g. ``EfficientNet_B0_Weights.DEFAULT``), the
    exact preprocessing the pretrained model was trained with is reused,
    which matters for transfer learning accuracy. Otherwise a sensible
    ImageNet-style default is used.
    """
    if weights is not None:
        return weights.transforms()

    return T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_dataloaders(
    train_dir: str | Path,
    test_dir: str | Path,
    transform: Callable,
    batch_size: int = 32,
    num_workers: int = 0,
) -> DataBundle:
    """Create train/test ``ImageFolder`` datasets and their dataloaders.

    Raises:
        FileNotFoundError: if ``train_dir`` or ``test_dir`` do not exist.
    """
    train_dir, test_dir = Path(train_dir), Path(test_dir)
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(test_dir, transform=transform)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return DataBundle(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        class_names=train_dataset.classes,
        class_to_idx=train_dataset.class_to_idx,
    )
