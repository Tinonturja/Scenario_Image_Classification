"""Training and evaluation loops.

This is a refactor of the ``train``/``test``/``engine`` functions from the
original notebook, with type hints, docstrings, and an added
``evaluate_with_report`` helper for a proper classification report and
confusion matrix (not present in the original exploratory notebook).
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def accuracy_fn(y_pred_logits: torch.Tensor, y_true: torch.Tensor) -> int:
    """Number of correct predictions in a batch (not normalized)."""
    pred_label = torch.argmax(y_pred_logits, dim=1)
    return int((pred_label == y_true).sum().item())


def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.to(device)
    model.train()

    total_loss, total_correct, num_samples = 0.0, 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        y_logits = model(X)
        loss = loss_fn(y_logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += accuracy_fn(y_logits, y)
        num_samples += X.shape[0]

    return total_loss / len(dataloader), total_correct / num_samples


def evaluate_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one evaluation epoch. Returns (avg_loss, accuracy)."""
    model.to(device)
    model.eval()

    total_loss, total_correct, num_samples = 0.0, 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            loss = loss_fn(y_logits, y)

            total_loss += loss.item()
            total_correct += accuracy_fn(y_logits, y)
            num_samples += X.shape[0]

    return total_loss / len(dataloader), total_correct / num_samples


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Train for ``epochs`` epochs, returning a history dict of metrics."""
    results: Dict[str, List[float]] = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = evaluate_step(model, test_dataloader, loss_fn, device)

        if verbose:
            print(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.3f} train_acc={train_acc:.3f} | "
                f"test_loss={test_loss:.3f} test_acc={test_acc:.3f}"
            )

        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        results["train_acc"].append(train_acc)
        results["test_acc"].append(test_acc)

    return results


def evaluate_with_report(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: List[str],
    device: torch.device,
):
    """Compute a full classification report and confusion matrix.

    This goes beyond the original notebook (which only tracked running
    accuracy) and is what you actually want to know before calling a model
    "87% accurate" — per-class precision/recall, not just an aggregate.

    Returns:
        (report_dict, confusion_matrix) where ``report_dict`` is the output
        of ``sklearn.metrics.classification_report(..., output_dict=True)``.
    """
    from sklearn.metrics import classification_report, confusion_matrix

    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            y_logits = model(X)
            preds = torch.argmax(y_logits, dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    return report, cm
