# Skin Lesion Segmentation

Skin lesion segmentation using Enhanced Attention UNet for the ISIC 2018 challenge.

## Model Performance

| Model | Parameters | Validation Dice | Test Dice (avg) |
|-------|------------|-----------------|-----------------|
| EnhancedAttentionUNet | 33M | 0.9045 | 0.9648 |

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

The model achieves:
- **Validation Dice**: 0.9045
- **Average Test Dice**: 0.9648 (on 10 random test images)

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