# Skin Lesion Segmentation

Skin lesion segmentation using Enhanced Attention UNet for the ISIC 2018 challenge.

## Quick Reviewer View

If you are reviewing this project in 2-3 minutes, check these first:

1. `README.md` (scope + reported metrics)
2. `docs/TRAINING_ARTIFACTS.md` (how logs/results are generated and published)
3. `eval.py` (quantitative evaluation + per-sample export)
4. `infer.py` (practical inference and qualitative overlays)

## Model Performance (Curated for Review)

The current published result uses the earliest stable 40-epoch run from today (`20260411_200313`, checkpoint epoch 40).

These are dataset-level averages from full split evaluation (not cherry-picked single images):

| Split | Dice | IoU | HD95 |
|-------|------|-----|------|
| Official Validation | 0.8774 | 0.8089 | 17.4679 |
| Test | 0.8796 | 0.8116 | 17.6414 |

Std (test split): Dice `0.1666`, HD95 `42.8784`.

Note for reviewers: earlier high numbers like `0.9648` came from individual sample rows in per-sample CSV, not overall test mean.

## Project Structure

```
skin-lesion-segmentation/
├── configs/
│   └── train_isic.yaml      # Training configuration
├── data/
│   └── dataset.py           # Dataset and data loading
├── models/
│   ├── components/          # Model architectures
│   │   ├── unet.py
│   │   ├── attention_unet.py
│   │   └── enhanced_attention_unet.py
│   └── losses/              # Loss functions
├── saved_models/            # Pre-trained weights
│   └── enhanced_attention_unet_best.pth
├── train.py                 # Training script
├── eval.py                  # Evaluation script
├── infer.py                 # Inference script
├── evaluate.py              # Model evaluation utilities
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

Download the ISIC 2018 dataset from the official website:

- **ISIC Archive**: https://www.isic-archive.com/
- **Download**: https://www.isic-archive.com/#top

Organize the dataset as follows:

```
ISIC2018/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

## Training

```bash
python train.py --config configs/train_isic.yaml --epochs 100
```

## Inference

```bash
python infer.py --image_path path/to/image.jpg
```

## Evaluation

```bash
python eval.py --checkpoint saved_models/enhanced_attention_unet_best.pth
```

## Results

Published artifact paths for advisor review:
- `logs/enhanced_attention_unet/20260411_200313/train.log`
- `logs/enhanced_attention_unet/20260411_200313/enhanced_attention_unet_20260411_200313_history.csv`
- `logs/enhanced_attention_unet/20260411_200313/enhanced_attention_unet_20260411_200313_curves.png`
- `results/compare_old_on_official_val/evaluation_summary_20260411_222522.json`
- `results/compare_old_on_test/evaluation_summary_20260411_223457.json`

This project reports conservative, reproducible metrics and keeps Dice + HD95 together to avoid over-claiming quality.

## Training/Evaluation Artifact Policy

To help advisors inspect experiment process (not just final numbers), this repo supports publishing curated artifacts:

- training log (`train.log`)
- evaluation summary JSON
- per-sample metrics CSV
- representative visualization PNGs

See `docs/TRAINING_ARTIFACTS.md` for the exact workflow.

Use helper script to generate an artifact index once logs/results exist:

```bash
python publish_artifacts.py
```

## Why HD95 may look worse while Dice improves

- Dice measures overlap and can improve steadily even when boundary outliers remain.
- HD95 is sensitive to boundary outliers; a few hard samples can keep it high.
- In this project, HD95 is logged every validation epoch and sampled in training batches.
- Use both metrics together when judging run quality.

## Citation

```bibtex
@article{ronneberger2015unet,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  journal={MICCAI},
  year={2015}
}

@article{woo2018cbam,
  title={CBAM: Convolutional Block Attention Module},
  author={Woo, Sanghyun and Park, Joon-Young and Kweon, In-So},
  journal={ECCV},
  year={2018}
}
```

## License

MIT
