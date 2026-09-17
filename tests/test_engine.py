import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scene_classifier.engine import accuracy_fn, evaluate_step, evaluate_with_report, train_step


def _toy_dataloader(n=16, num_classes=3):
    X = torch.randn(n, 3, 8, 8)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=4)


def _toy_model(num_classes=3):
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, num_classes))


def test_accuracy_fn_counts_correct_predictions():
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0]])
    labels = torch.tensor([0, 1, 1])  # last one is wrong
    assert accuracy_fn(logits, labels) == 2


def test_train_step_reduces_loss_direction_is_sane():
    torch.manual_seed(0)
    model = _toy_model()
    dataloader = _toy_dataloader()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    loss_before, _ = evaluate_step(model, dataloader, loss_fn, torch.device("cpu"))
    for _ in range(5):
        train_step(model, dataloader, loss_fn, optimizer, torch.device("cpu"))
    loss_after, _ = evaluate_step(model, dataloader, loss_fn, torch.device("cpu"))

    assert loss_after < loss_before


def test_evaluate_with_report_shapes():
    model = _toy_model(num_classes=3)
    dataloader = _toy_dataloader(n=12, num_classes=3)
    class_names = ["a", "b", "c"]

    report, cm = evaluate_with_report(model, dataloader, class_names, torch.device("cpu"))

    assert cm.shape == (3, 3)
    for cls in class_names:
        assert cls in report
