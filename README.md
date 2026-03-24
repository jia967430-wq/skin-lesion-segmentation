# skin-lesion-segmentation

皮肤病变分割模型 - 基于 U-Net + CBAM 注意力机制的 PyTorch 实现

## 项目背景

皮肤镜图像分割是皮肤病变分析的基础任务。本项目实现了两种分割模型：

- **UNet**：经典医学图像分割架构
- **UNet + CBAM**：在 UNet 基础上加入通道和空间注意力模块

在 ISIC 2018 皮肤病变分割数据集上进行了训练和验证。

## 项目结构

```
├── models/              # 模型定义
│   ├── components/      # 网络组件（UNet、CBAM等）
│   └── losses/          # 损失函数
├── data/                # 数据加载
├── configs/             # 训练配置
├── experiments/         # 消融实验脚本
├── utils/               # 工具函数
├── scripts/             # 数据准备
├── docs/                # 文档
└── train.py             # 训练入口
```

## 环境

```bash
pip install -r requirements.txt
```

- Python 3.8+
- PyTorch 2.0+
- albumentations

## 数据

本项目在 ISIC 2018 上训练，但数据加载模块支持按相同格式组织的其他分割数据集。

### 目录格式

```
your_data/
├── train/
│   ├── images/    # 图像 (jpg/png)
│   └── masks/     # 标注 (png, 单通道二值图)
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### ISIC 2018

1. 从官网下载：https://challenge.isic-archive.com/data/2018
2. 解压后运行 `python scripts/prepare_data.py --data_dir data/`

### 其他数据集

只要按上述目录格式组织，修改配置文件中的 `root_dir` 即可。

## 训练

```bash
# UNet baseline
python train.py --config configs/train_isic.yaml --model unet --epochs 50

# UNet + CBAM
python train.py --config configs/train_isic.yaml --model attention_unet --epochs 50
```

## 模型

### UNet

标准编码器-解码器结构，跳跃连接融合多尺度特征。

### UNet + CBAM

在 UNet 每个阶段加入 CBAM（Convolutional Block Attention Module）：

- 通道注意力：学习"关注哪些特征"
- 空间注意力：学习"关注哪些位置"

## 实验结果

ISIC 2018 测试集（n=331）：

| 模型 | Dice | IoU | Precision | Recall |
|------|------|-----|-----------|--------|
| UNet | 0.873 | 0.803 | 0.911 | 0.872 |
| UNet + CBAM | 0.883 | 0.818 | 0.910 | 0.888 |

## 扩展

如果想在其他分割任务上使用本项目：

1. 按上述格式准备数据
2. 修改 `configs/train_isic.yaml` 中的路径和类别数
3. 如有多类别分割，将 `out_channels` 改为类别数

## 引用

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
