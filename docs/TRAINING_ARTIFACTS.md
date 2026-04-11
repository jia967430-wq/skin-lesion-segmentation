# Training Artifacts and Reproducibility

This project can publish training/evaluation artifacts for advisor review.

Current status in this repository snapshot:

- No local `logs/`, `results/`, or `checkpoints/` directories were found.
- Model checkpoints are intentionally excluded by `.gitignore` (`*.pth`, `*.pt`).
- The current tracked repo is code-first and does not yet include exported experiment artifacts.

## How to Generate Publishable Artifacts

### 1) Training logs

Run training (example):

```bash
python train.py --config configs/train_isic.yaml --epochs 60
```

This creates a run folder similar to:

- `logs/<model_name>/<timestamp>/train.log`
- `logs/<model_name>/<timestamp>/tensorboard/`

### 2) Evaluation reports

Run evaluation after training:

```bash
python eval.py --checkpoint <BEST_CHECKPOINT_PATH> --dataset <ISIC_ROOT> --split test --output results --viz
```

This creates:

- `results/evaluation_summary_<timestamp>.json`
- `results/per_sample_results_<timestamp>.csv`
- `results/visualizations/*.png`

### 3) Publish selected artifacts (recommended)

Do not publish full checkpoints or raw image data. Publish a curated subset:

- one representative `train.log`
- one `evaluation_summary_*.json`
- one `per_sample_results_*.csv`
- 6-12 representative `results/visualizations/*.png`

## Suggested Additions to README

- report your latest run ID/timestamp
- link to summary JSON + per-sample CSV
- link to visualization examples
- state dataset split and threshold setting used in evaluation
