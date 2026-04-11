# Training Artifacts and Reproducibility

This project publishes curated training/evaluation artifacts for advisor review.

Current curated run in this repository snapshot:

- Run ID: `20260411_200313` (today's first stable 40-epoch run)
- Training log: `logs/enhanced_attention_unet/20260411_200313/train.log`
- History CSV: `logs/enhanced_attention_unet/20260411_200313/enhanced_attention_unet_20260411_200313_history.csv`
- Curves PNG: `logs/enhanced_attention_unet/20260411_200313/enhanced_attention_unet_20260411_200313_curves.png`
- Validation summary: `results/compare_old_on_official_val/evaluation_summary_20260411_222522.json`
- Validation per-sample: `results/compare_old_on_official_val/per_sample_results_20260411_222522.csv`
- Test summary: `results/compare_old_on_test/evaluation_summary_20260411_223457.json`
- Test per-sample: `results/compare_old_on_test/per_sample_results_20260411_223457.csv`

Checkpoint files are intentionally excluded by `.gitignore` (`*.pth`, `*.pt`).

## How to Generate Publishable Artifacts

### 1) Training logs

Run training (example):

```bash
python train.py --config configs/train_isic_tonight.yaml --epochs 70
```

This creates a run folder similar to:

- `logs/<model_name>/<timestamp>/train.log`
- `logs/<model_name>/<timestamp>/tensorboard/`

### 2) Evaluation reports

Run evaluation after training:

```bash
python eval.py --checkpoint <BEST_CHECKPOINT_PATH> --dataset <ISIC_ROOT> --split test --output results/tonight_final_eval --viz
```

This creates:

- `results/evaluation_summary_<timestamp>.json`
- `results/per_sample_results_<timestamp>.csv`
- `results/visualizations/*.png`

### 3) Publish selected artifacts (recommended)

Do not publish full checkpoints or raw image data. Publish a curated subset:

- one representative `train.log`
- one training history CSV and curves PNG
- one `evaluation_summary_*.json`
- one `per_sample_results_*.csv`
- 6-12 representative `results/.../visualizations/*.png`

## Suggested Additions to README

- report your latest run ID/timestamp
- link to summary JSON + per-sample CSV
- link to visualization examples
- state dataset split and threshold setting used in evaluation
- include Dice and HD95 together for quality judgment
