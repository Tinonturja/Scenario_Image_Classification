# Scene Classification with EfficientNet-B0

Transfer-learning image classifier that sorts natural scene photos into six
categories — **buildings, forest, glacier, mountain, sea, street** — using a
pretrained EfficientNet-B0 backbone with a fine-tuned classifier head.

## Results

Trained for 5 epochs with a frozen EfficientNet-B0 backbone (ImageNet
weights) and a new linear classifier head, Adam optimizer, lr=1e-4.

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|-------|-----------|-----------|-----------|----------|
| 0 | 1.231 | 66.3% | 0.824 | 83.8% |
| 1 | 0.729 | 82.9% | 0.590 | 85.7% |
| 2 | 0.577 | 84.8% | 0.481 | 86.8% |
| 3 | 0.512 | 85.1% | 0.434 | 86.9% |
| 4 | 0.462 | 86.2% | 0.404 | **87.3%** |

![Training curves](results/training_curves.png)

Per-class breakdown on the test set:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| buildings | 0.892 | 0.865 | 0.878 |
| forest | 0.981 | 0.979 | 0.980 |
| glacier | 0.814 | 0.805 | 0.809 |
| mountain | 0.800 | 0.792 | 0.796 |
| sea | 0.877 | 0.892 | 0.884 |
| street | 0.890 | 0.918 | 0.904 |

`glacier` and `mountain` are the weakest classes, which lines up with the
intuition that they're the two most visually similar categories in this
dataset. These numbers are the exact values logged during a full end-to-end
training run of this refactored code (see `results/history.json`); the plot
above is generated directly from that file.

**On the codebase, not just the model:** `src/scene_classifier/engine.py`
also adds `evaluate_with_report()`, which computes a full per-class
precision/recall/F1 breakdown and confusion matrix — the original notebook
only tracked aggregate running accuracy, which hides whether the model is
systematically confusing two similar classes (e.g. `sea` vs. `glacier`).
Run `python -m scene_classifier.train` end-to-end to generate that report
for the current model.

### Example predictions

![Example predictions](assets/prediction_examples.png)

Real inferences from `results/model.pth` on the three personal photos in
`assets/examples/` — actual model output, not mockups. Regenerate after
training with:

```bash
python scripts/make_prediction_demo.py
```

## Dataset

The [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
(~14k training / ~3k test images across the six classes above). It is not
committed to this repository — download it from Kaggle and place it as:

```
seg_train/<class_name>/*.jpg
seg_test/<class_name>/*.jpg
```

## Project structure

```
.
├── src/scene_classifier/
│   ├── data.py      # dataset/dataloader construction + transforms
│   ├── model.py      # EfficientNet-B0 model builder
│   ├── engine.py      # train/eval loops + classification report
│   ├── train.py      # CLI entry point that runs the full pipeline
│   ├── predict.py    # single-image inference
│   └── utils.py      # seeding, device selection, checkpoint I/O, plotting
├── tests/             # unit tests (no dataset download required)
├── notebooks/         # original exploratory notebook, kept for provenance
└── results/           # training history, curves, classification report
```

This mirrors the logic of the original exploratory notebook
(`notebooks/01_efficientnet_b0_transfer_learning.ipynb`) but split into
tested, reusable modules instead of one linear notebook.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

## Usage

Train:

```bash
python -m scene_classifier.train --train-dir seg_train --test-dir seg_test --epochs 5
```

Predict on a single image:

```bash
python -m scene_classifier.predict path/to/image.jpg --checkpoint results/model.pth
```

`assets/examples/` has three downsized personal photos (resized copies of
the images used for the prediction demo in the original notebook) you can
use to try this without any dataset download:

```bash
python -m scene_classifier.predict assets/examples/example_1.jpg --checkpoint results/model.pth
```

Run tests:

```bash
pytest -v
```

## Notes on the refactor

The original notebook is preserved under `notebooks/` unchanged. The
`src/` package is a clean-room reimplementation of the same training logic:
same architecture, same hyperparameters, same data pipeline — restructured
into modules with type hints, docstrings, and a test suite, plus the added
per-class evaluation report described above.

The results table above is from a full end-to-end run of this refactored
code (`python -m scene_classifier.train`, 5 epochs, Apple Silicon `mps`
backend), not copied from the original notebook. It lands at 87.3% test
accuracy, close to the original notebook's 87.1% on the same architecture
and hyperparameters — the small difference is expected from a different
run (hardware, weight initialization) rather than a code discrepancy. The
unit tests in `tests/` separately validate the refactored code
(dataloaders, training step, evaluation, model shapes) against a synthetic
dataset without needing the real data or pretrained weights at all.

## License

MIT — see [LICENSE](LICENSE).
