"""CLI entry point that wires data, model, and engine together.

Example:
    python -m scene_classifier.train \\
        --train-dir seg_train --test-dir seg_test \\
        --epochs 5 --batch-size 32 --lr 1e-4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from scene_classifier.data import build_transforms, create_dataloaders
from scene_classifier.engine import evaluate_with_report, train
from scene_classifier.model import build_model
from scene_classifier.utils import get_device, plot_curves, save_checkpoint, save_history, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the EfficientNet-B0 scene classifier.")
    parser.add_argument("--train-dir", type=str, default="seg_train")
    parser.add_argument("--test-dir", type=str, default="seg_test")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    model, weights = build_model(num_classes=6, freeze_backbone=args.freeze_backbone)
    transform = build_transforms(weights)

    data = create_dataloaders(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Classes: {data.class_names}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = train(
        model=model,
        train_dataloader=data.train_dataloader,
        test_dataloader=data.test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device,
    )

    output_dir = Path(args.output_dir)
    save_history(history, output_dir / "history.json")
    plot_curves(history, save_path=output_dir / "training_curves.png")
    save_checkpoint(model, output_dir / "model.pth", extra={"class_names": data.class_names})

    report, cm = evaluate_with_report(model, data.test_dataloader, data.class_names, device)
    print("Per-class results:")
    for cls in data.class_names:
        r = report[cls]
        print(
            f"  {cls:12s} precision={r['precision']:.3f} "
            f"recall={r['recall']:.3f} f1={r['f1-score']:.3f}"
        )
    print(f"Overall accuracy: {report['accuracy']:.3f}")

    import json

    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
