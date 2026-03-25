"""
Demo inference script - 对示例图片进行分割
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, '.')

from models import UNet, AttentionUNet
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_model(ckpt_path, model_type='attention_unet'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    if model_type == 'attention_unet':
        model = AttentionUNet(in_channels=3, out_channels=1, base_filters=64)
    else:
        model = UNet(in_channels=3, out_channels=1, base_filters=64)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    return model, device

def predict(model, image_path, device, size=(224, 224)):
    image = np.array(Image.open(image_path).convert('RGB'))
    
    transform = A.Compose([
        A.Resize(size[0], size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    augmented = transform(image=image)
    input_tensor = augmented['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
    
    pred_binary = (pred > 0.5).astype(np.uint8) * 255
    
    return image, pred, pred_binary

def visualize(image, pred, pred_binary, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(pred, cmap='hot')
    axes[1].set_title('Prediction (Probability)')
    axes[1].axis('off')
    
    axes[2].imshow(pred_binary, cmap='gray')
    axes[2].set_title('Segmentation Result')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')

def main():
    print('='*50)
    print('Skin Lesion Segmentation Demo')
    print('='*50)
    
    # 找checkpoint
    ckpt_dir = Path('checkpoints')
    if not ckpt_dir.exists():
        # 尝试从D盘找
        ckpt_dir = Path('D:/medseg-data/checkpoints')
    
    ckpts = sorted(ckpt_dir.glob('attention_unet_*_best.pth'))
    if not ckpts:
        ckpts = sorted(ckpt_dir.glob('*best.pth'))
    
    if not ckpts:
        print('No checkpoint found!')
        return
    
    print(f'Using: {ckpts[-1].name}')
    model, device = load_model(ckpts[-1], 'attention_unet')
    print(f'Device: {device}')
    
    # 找测试图片
    samples_dir = Path('samples')
    if not samples_dir.exists():
        samples_dir = Path('D:/medseg-data/ISIC2018/test/images')
    
    images = list(samples_dir.glob('*.jpg'))[:5]
    
    if not images:
        print('No images found!')
        return
    
    # 创建输出目录
    output_dir = Path('demo_results')
    output_dir.mkdir(exist_ok=True)
    
    # 推理
    for img_path in images:
        print(f'\\nProcessing: {img_path.name}')
        image, pred, pred_binary = predict(model, img_path, device)
        
        # 保存结果
        save_path = output_dir / f'result_{img_path.stem}.png'
        visualize(image, pred, pred_binary, save_path)
    
    print(f'\\nDone! Results saved to: {output_dir}/')
    print('Open the PNG files to see the segmentation results.')

if __name__ == '__main__':
    main()
