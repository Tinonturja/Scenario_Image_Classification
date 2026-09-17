import torch

from scene_classifier.model import build_model, count_trainable_parameters

# pretrained=False everywhere here: these tests check architecture/shape
# logic, not transfer-learning accuracy, and shouldn't depend on network
# access to download.pytorch.org.


def test_build_model_output_shape():
    model, weights = build_model(num_classes=6, pretrained=False)
    assert weights is None
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 6)


def test_build_model_freezes_backbone_by_default():
    model, _ = build_model(num_classes=6, freeze_backbone=True, pretrained=False)
    backbone_params = list(model.features.parameters())
    assert all(not p.requires_grad for p in backbone_params)

    head_params = list(model.classifier.parameters())
    assert all(p.requires_grad for p in head_params)


def test_build_model_can_unfreeze_backbone():
    model, _ = build_model(num_classes=6, freeze_backbone=False, pretrained=False)
    backbone_params = list(model.features.parameters())
    assert all(p.requires_grad for p in backbone_params)


def test_count_trainable_parameters_matches_frozen_state():
    model, _ = build_model(num_classes=6, freeze_backbone=True, pretrained=False)
    trainable = count_trainable_parameters(model)
    expected = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    assert trainable == expected
