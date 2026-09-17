"""Run inference with a trained model on a single image."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple

import torch
from PIL import Image
from torch import nn


def predict_image(
    model: nn.Module,
    image_path: str | Path,
    class_names: List[str],
    transform: Callable,
    device: torch.device,
) -> Tuple[str, float]:
    """Predict the class of a single image.

    Returns:
        (predicted_class_name, confidence)
    """
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    model.to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs.max().item())

    return class_names[pred_idx], confidence


if __name__ == "__main__":
    import argparse

    from scene_classifier.data import build_transforms
    from scene_classifier.model import build_model
    from scene_classifier.utils import get_device, load_checkpoint

    parser = argparse.ArgumentParser(description="Predict the scene class of an image.")
    parser.add_argument("image", type=str, help="Path to an image file")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to a saved checkpoint (.pth)"
    )
    parser.add_argument(
        "--class-names",
        type=str,
        nargs="+",
        default=["buildings", "forest", "glacier", "mountain", "sea", "street"],
    )
    args = parser.parse_args()

    device = get_device()
    model, weights = build_model(num_classes=len(args.class_names))
    model = load_checkpoint(model, args.checkpoint, map_location=str(device))
    transform = build_transforms(weights)

    pred_class, confidence = predict_image(model, args.image, args.class_names, transform, device)
    print(f"Prediction: {pred_class} (confidence: {confidence:.3f})")
