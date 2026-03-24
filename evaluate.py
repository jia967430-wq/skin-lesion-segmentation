"""
Test Set Evaluation

Evaluates model on full test set and generates comprehensive report.
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
import sys

sys.path.insert(0, '.')
from models import UNet, AttentionUNet
from data.dataset import MedicalSegmentationDataset, get_test_transforms
from torch.utils.data import DataLoader

def calculate_metrics(pred, gt):
    """Calculate segmentation metrics"""
    # Ensure binary
    pred = (pred > 0.5).astype(np.uint8)
    gt = (gt > 0.5).astype(np.uint8)
    
    # Calculate metrics
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    tn = np.logical_and(pred == 0, gt == 0).sum()
    
    # Dice
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    
    # IoU
    iou = tp / (tp + fp + fn + 1e-8)
    
    # Precision, Recall, F1
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }

def evaluate_model(model, dataloader, device):
    """Evaluate model on dataset"""
    model.eval()
    
    all_metrics = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['image'].to(device)
            masks = batch['mask']
            
            # Predict
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            
            # Calculate metrics for each sample
            for i in range(images.shape[0]):
                pred = preds[i, 0]
                gt = masks[i].numpy()
                
                metrics = calculate_metrics(pred, gt)
                all_metrics.append(metrics)
    
    # Calculate averages
    avg_metrics = {
        'dice': np.mean([m['dice'] for m in all_metrics]),
        'iou': np.mean([m['iou'] for m in all_metrics]),
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
        'f1': np.mean([m['f1'] for m in all_metrics])
    }
    
    # Standard deviations
    std_metrics = {
        'dice_std': np.std([m['dice'] for m in all_metrics]),
        'iou_std': np.std([m['iou'] for m in all_metrics])
    }
    
    return avg_metrics, std_metrics, all_metrics

def run_evaluation():
    """Run full test set evaluation"""
    print("="*60)
    print("Test Set Evaluation")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create dataset
    dataset = MedicalSegmentationDataset(
        root_dir='data/ISIC2018',
        split='test',
        transform=get_test_transforms((224, 224)),
        image_size=(224, 224)
    )
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    print(f"Test samples: {len(dataset)}")
    
    # Results storage
    results = {}
    
    # Evaluate each model
    ckpt_dir = Path('checkpoints')
    
    for model_type, pattern in [('UNet', 'unet_*_best.pth'), ('AttentionUNet', 'attention_unet_*_best.pth')]:
        ckpts = sorted(ckpt_dir.glob(pattern))
        if not ckpts:
            print(f"No {model_type} checkpoint found")
            continue
        
        ckpt_path = ckpts[-1]
        print(f"\nEvaluating {model_type}: {ckpt_path.name}")
        
        # Load model
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        if model_type == 'UNet':
            model = UNet(in_channels=3, out_channels=1, base_filters=64)
        else:
            model = AttentionUNet(in_channels=3, out_channels=1, base_filters=64)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Evaluate
        avg_metrics, std_metrics, all_metrics = evaluate_model(model, dataloader, device)
        
        results[model_type] = {
            'avg': avg_metrics,
            'std': std_metrics,
            'checkpoint': ckpt_path.name,
            'val_dice': checkpoint.get('best_dice', 'N/A')
        }
        
        print(f"  Val Dice: {checkpoint.get('best_dice', 'N/A'):.4f}")
        print(f"  Test Dice: {avg_metrics['dice']:.4f} +/- {std_metrics['dice_std']:.4f}")
        print(f"  Test IoU: {avg_metrics['iou']:.4f} +/- {std_metrics['iou_std']:.4f}")
        print(f"  Precision: {avg_metrics['precision']:.4f}")
        print(f"  Recall: {avg_metrics['recall']:.4f}")
        print(f"  F1: {avg_metrics['f1']:.4f}")
    
    # Save results
    output_path = Path('results/evaluation_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return results

if __name__ == '__main__':
    run_evaluation()
