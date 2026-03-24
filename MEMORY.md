# 皮肤病变分割项目 - 状态记录

## 当前状态

**项目位置**: C:\Users\Diavolo\Documents\skin-lesion-segmentation  
**数据位置**: D:\medseg-data\ISIC2018  
**GitHub仓库**: https://github.com/jia967430-wq/skin-lesion-segmentation (尚未push)

## 已完成

- ✅ 代码重构完成，去掉Mamba相关命名
- ✅ README重写，用词准确无AI感
- ✅ 项目结构整理完毕
- ✅ git init + commit完成
- ✅ GitHub仓库已创建

## 问题

- ❌ 训练好的模型(checkpoints/)被误删，无法恢复
- ❌ 未完成GitHub push

## 需要重新训练

之前训练结果：
- UNet baseline: Dice 0.8729
- UNet + CBAM: Dice 0.8832

训练命令：
```bash
cd C:\Users\Diavolo\Documents\skin-lesion-segmentation
# 需要先安装依赖
pip install torch torchvision albumentations segmentation-models-pytorch

# 训练 (数据在D盘)
python train.py --config configs/train_isic.yaml --model unet --epochs 30
python train.py --config configs/train_isic.yaml --model attention_unet --epochs 30
```

## 数据集路径

配置文件中root_dir需要改为：`D:/medseg-data/ISIC2018`

## 待完成

1. 重新训练模型
2. 评估并更新README中的结果
3. git add + commit + push到GitHub
