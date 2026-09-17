"""Generate a grid of real model predictions on sample images for the README.

Run after training (needs results/model.pth):
    python scripts/make_prediction_demo.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from scene_classifier.data import build_transforms
from scene_classifier.model import build_model
from scene_classifier.utils import get_device, load_checkpoint

CHECKPOINT = Path("results/model.pth")
EXAMPLES_DIR = Path("assets/examples")
OUTPUT = Path("assets/prediction_examples.png")
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]


def main() -> None:
    device = get_device()

    model, weights = build_model(num_classes=len(CLASS_NAMES))
    model = load_checkpoint(model, CHECKPOINT, map_location=str(device))
    model.to(device)
    model.eval()

    transform = build_transforms(weights)

    image_paths = sorted(EXAMPLES_DIR.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No example images found in {EXAMPLES_DIR}")

    n = len(image_paths)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, path in zip(axes, image_paths):
        image = Image.open(path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs.max().item())

        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"Pred: {CLASS_NAMES[pred_idx]} ({confidence:.1%})", fontsize=12)

    fig.suptitle(
        "Real predictions from results/model.pth on assets/examples/ photos",
        fontsize=12,
    )
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=100, bbox_inches="tight")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
