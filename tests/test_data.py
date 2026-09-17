import pytest
import torch

from scene_classifier.data import build_transforms, create_dataloaders


def test_build_transforms_default_produces_correct_tensor_shape():
    from PIL import Image

    transform = build_transforms(weights=None)
    dummy_image = Image.new("RGB", (300, 400))
    tensor = transform(dummy_image)
    assert tensor.shape == (3, 224, 224)
    assert isinstance(tensor, torch.Tensor)


def test_create_dataloaders_raises_on_missing_dir(tmp_path):
    transform = build_transforms(weights=None)
    with pytest.raises(FileNotFoundError):
        create_dataloaders(
            train_dir=tmp_path / "does_not_exist_train",
            test_dir=tmp_path / "does_not_exist_test",
            transform=transform,
        )


def test_create_dataloaders_with_tiny_synthetic_dataset(tmp_path):
    """Build a minimal ImageFolder-style dataset on disk and check the
    dataloaders come back with the expected classes and batch shapes.
    """
    from PIL import Image

    for split in ["train", "test"]:
        for cls in ["forest", "sea"]:
            class_dir = tmp_path / split / cls
            class_dir.mkdir(parents=True)
            for i in range(3):
                Image.new("RGB", (50, 50)).save(class_dir / f"{i}.jpg")

    transform = build_transforms(weights=None)
    data = create_dataloaders(
        train_dir=tmp_path / "train",
        test_dir=tmp_path / "test",
        transform=transform,
        batch_size=2,
    )

    assert sorted(data.class_names) == ["forest", "sea"]
    X, y = next(iter(data.train_dataloader))
    assert X.shape[1:] == (3, 224, 224)
    assert y.shape[0] == X.shape[0]
