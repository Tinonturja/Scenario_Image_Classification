"""Model construction: EfficientNet-B0 with a frozen backbone and a
fresh classifier head, sized for the number of scene classes.
"""

from __future__ import annotations

import torchvision
from torch import nn


def build_model(
    num_classes: int,
    freeze_backbone: bool = True,
    dropout: float = 0.2,
    pretrained: bool = True,
) -> tuple[nn.Module, torchvision.models.WeightsEnum | None]:
    """Build an EfficientNet-B0 transfer-learning model.

    Args:
        num_classes: number of output classes for the new classifier head.
        freeze_backbone: if True, the pretrained feature extractor's
            weights are frozen and only the new classifier head is trained.
        dropout: dropout probability used in the classifier head.
        pretrained: if True (default), download and use ImageNet-pretrained
            weights. Set to False to build the architecture only, with no
            network access — used by the test suite so it doesn't depend on
            being able to reach download.pytorch.org.

    Returns:
        (model, weights) — ``weights`` is ``None`` when ``pretrained=False``,
        otherwise its ``.transforms()`` method is needed to build a matching
        preprocessing pipeline.
    """
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = torchvision.models.efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features  # 1280 for EfficientNet-B0
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=in_features, out_features=num_classes),
    )

    return model, weights


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
