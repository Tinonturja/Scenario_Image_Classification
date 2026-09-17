"""Small shared utilities: reproducibility, device selection, checkpointing,
and plotting.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Pick the best available device: CUDA > Apple MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(model: nn.Module, path: str | Path, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model: nn.Module, path: str | Path, map_location: str = "cpu") -> nn.Module:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def save_history(history: Dict[str, List[float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def plot_curves(history: Dict[str, List[float]], save_path: str | Path | None = None):
    """Plot train/test loss and accuracy curves side by side."""
    import matplotlib.pyplot as plt

    epochs = range(len(history["train_loss"]))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["test_loss"], label="Test Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Accuracy")
    axes[1].plot(epochs, history["test_acc"], label="Test Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig
