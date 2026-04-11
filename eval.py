"""
Evaluation Script for Medical Image Segmentation

Evaluates trained models on test set and generates:
- Quantitative metrics (Dice, IoU, Precision, Recall, F1)
- Qualitative results (visualizations)
- Per-sample results CSV
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import binary_erosion, distance_transform_edt

from models import UNet, AttentionUNet, AttentionUNetLite, EnhancedAttentionUNet
from data.dataset import MedicalSegmentationDataset, get_test_transforms


def calculate_metrics(pred, target, threshold=0.5, smooth=1e-6):
    """Calculate segmentation metrics with proper handling for edge cases."""
    pred = pred.float()
    target = target.float()
    
    # Handle different tensor shapes
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()
    
    intersection = (pred_binary * target_binary).sum()
    pred_sum = pred_binary.sum()
    target_sum = target_binary.sum()
    
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    dice = torch.clamp(dice, 0.0, 1.0)
    
    union = pred_sum + target_sum - intersection
    iou = (intersection + smooth) / (union + smooth)
    iou = torch.clamp(iou, 0.0, 1.0)
    
    tp = intersection
    fp = pred_sum - intersection
    fn = target_sum - intersection
    
    precision = (tp + smooth) / (tp + fp + smooth)
    precision = torch.clamp(precision, 0.0, 1.0)
    
    recall = (tp + smooth) / (tp + fn + smooth)
    recall = torch.clamp(recall, 0.0, 1.0)
    
    f1 = 2 * precision * recall / (precision + recall + smooth)
    f1 = torch.clamp(f1, 0.0, 1.0)
    
    correct = (pred_binary == target_binary).sum()
    accuracy = correct / target_binary.numel()
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'accuracy': accuracy.item()
    }


def calculate_hd95(pred, target, threshold=0.5):
    pred = pred.float()
    target = target.float()
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    pred_b = (pred > threshold).cpu().numpy().astype(bool)
    tgt_b = (target > 0.5).cpu().numpy().astype(bool)
    vals = []
    for p, g in zip(pred_b, tgt_b):
        h, w = p.shape[-2], p.shape[-1]
        max_dist = float(np.hypot(h, w))
        if (not p.any()) and (not g.any()):
            vals.append(0.0)
            continue
        if (not p.any()) or (not g.any()):
            vals.append(max_dist)
            continue
        p_s = p ^ binary_erosion(p)
        g_s = g ^ binary_erosion(g)
        d_g = distance_transform_edt(~g_s)
        d_p = distance_transform_edt(~p_s)
        all_d = np.concatenate([d_g[p_s], d_p[g_s]])
        vals.append(float(np.percentile(all_d, 95)))
    finite = [v for v in vals if np.isfinite(v)]
    if not finite:
        return float('nan')
    return float(np.mean(finite))


def find_optimal_threshold(preds, targets, thresholds=None):
    """Find optimal binarization threshold by grid search."""
    if thresholds is None:
        thresholds = [i * 0.05 for i in range(21)]  # 0.0 to 1.0 with step 0.05
    
    best_threshold = 0.5
    best_dice = 0.0
    
    results = []
    for thresh in thresholds:
        dice_scores = []
        for pred, target in zip(preds, targets):
            metrics = calculate_metrics(pred, target, threshold=thresh)
            dice_scores.append(metrics['dice'])
        
        mean_dice = np.mean(dice_scores)
        results.append({'threshold': thresh, 'dice': mean_dice})
        
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_threshold = thresh
    
    return best_threshold, best_dice, results


def visualize_results(image, mask, pred, save_path, title=''):
    """Visualize segmentation results"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Image
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)
    
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Ground truth
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy().squeeze()
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy().squeeze()
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Overlay
    overlay = image.copy()
    if len(overlay.shape) == 2:
        overlay = np.stack([overlay] * 3, axis=-1)
    
    mask_bool = mask > 0.5
    pred_bool = pred > 0.5
    
    overlay[mask_bool & pred_bool] = [0, 1, 0]  # TP - Green
    overlay[mask_bool & ~pred_bool] = [1, 0, 0]  # FN - Red (missed)
    overlay[~mask_bool & pred_bool] = [0, 0, 1]  # FP - Blue (false positive)
    
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay (G=TP, R=FN, B=FP)')
    axes[3].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


class Evaluator:
    """Model evaluator"""
    
    def __init__(self, checkpoint_path, config=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = config or self.checkpoint.get('config', {})
        
        # Create model
        self.model = self._create_model()
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"[INFO] Loaded model from: {checkpoint_path}")
        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Epoch: {self.checkpoint.get('epoch', 'N/A')}")
        print(f"[INFO] Best Dice: {self.checkpoint.get('best_dice', 'N/A'):.4f}")
    
    def _create_model(self):
        """Create model from checkpoint config"""
        model_name = self.config.get('model', {}).get('name', 'unet')
        
        if model_name == 'attention_unet':
            return AttentionUNet(
                in_channels=self.config.get('data', {}).get('in_channels', 3),
                out_channels=self.config.get('data', {}).get('out_channels', 1),
                base_filters=self.config.get('model', {}).get('base_filters', 64)
            )
        elif model_name == 'attention_unet_lite':
            return AttentionUNetLite(
                in_channels=self.config.get('data', {}).get('in_channels', 3),
                out_channels=self.config.get('data', {}).get('out_channels', 1),
                base_filters=self.config.get('model', {}).get('base_filters', 32)
            )
        elif model_name == 'enhanced_attention_unet':
            return EnhancedAttentionUNet(
                in_channels=self.config.get('data', {}).get('in_channels', 3),
                out_channels=self.config.get('data', {}).get('out_channels', 1),
                base_filters=self.config.get('model', {}).get('base_filters', 64),
                deep_supervision=self.config.get('model', {}).get('deep_supervision', False)
            )
        else:
            return UNet(
                in_channels=self.config.get('data', {}).get('in_channels', 3),
                out_channels=self.config.get('data', {}).get('out_channels', 1),
                base_filters=self.config.get('model', {}).get('base_filters', 64)
            )
    
    def evaluate(self, dataset_path, split='test', save_viz=True, viz_dir='results/visualizations', find_threshold=False):
        """Evaluate model on dataset"""
        # Create dataloader
        image_size = tuple(self.config.get('data', {}).get('image_size', [224, 224]))
        
        dataset = MedicalSegmentationDataset(
            root_dir=dataset_path,
            split=split,
            transform=get_test_transforms(image_size),
            image_size=image_size
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )
        
        print(f"[INFO] Evaluating on {len(dataset)} images...")
        
        # Storage for results
        all_metrics = []
        per_sample_results = []
        all_preds = []
        all_targets = []
        
        # Create visualization directory
        viz_dir = Path(viz_dir)
        if save_viz:
            viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluate
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader)):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                filename = batch['filename'][0]
                
                # Forward pass
                outputs = self.model(images)
                preds = torch.sigmoid(outputs)
                
                # Calculate metrics
                metrics = calculate_metrics(preds.squeeze(), masks.squeeze())
                metrics['hd95'] = calculate_hd95(preds, masks)
                all_metrics.append(metrics)
                
                per_sample_results.append({
                    'filename': filename,
                    **metrics
                })
                
                # Store for threshold search
                if find_threshold:
                    all_preds.append(preds.squeeze().cpu())
                    all_targets.append(masks.squeeze().cpu())
                
                # Save visualization
                if save_viz and idx < 20:  # Save first 20 visualizations
                    save_path = viz_dir / f'{Path(filename).stem}_viz.png'
                    visualize_results(
                        images.squeeze(),
                        masks.squeeze(),
                        preds.squeeze(),
                        save_path,
                        title=f"{filename}"
                    )
        
        # Find optimal threshold if requested
        threshold_result = None
        if find_threshold and all_preds:
            best_thresh, best_dice, threshold_result = find_optimal_threshold(all_preds, all_targets)
            print(f"[INFO] Optimal threshold: {best_thresh:.2f} (Dice: {best_dice:.4f})")
            
            # Recalculate metrics with optimal threshold
            all_metrics = []
            for pred, target in zip(all_preds, all_targets):
                metrics = calculate_metrics(pred, target, threshold=best_thresh)
                metrics['hd95'] = calculate_hd95(pred.unsqueeze(0), target.unsqueeze(0), threshold=best_thresh)
                all_metrics.append(metrics)
            
            per_sample_results = []
            for idx, (pred, target) in enumerate(zip(all_preds, all_targets)):
                metrics = calculate_metrics(pred, target, threshold=best_thresh)
                metrics['hd95'] = calculate_hd95(pred.unsqueeze(0), target.unsqueeze(0), threshold=best_thresh)
                per_sample_results.append({
                    'filename': dataset.image_files[idx],
                    **metrics
                })
        
        # Aggregate metrics
        avg_metrics = {
            k: np.mean([m[k] for m in all_metrics])
            for k in all_metrics[0].keys()
        }
        
        std_metrics = {
            k: np.std([m[k] for m in all_metrics])
            for k in all_metrics[0].keys()
        }
        
        return avg_metrics, std_metrics, per_sample_results, threshold_result
    
    def save_results(self, results, output_dir='results'):
        """Save evaluation results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save summary
        summary = {
            'timestamp': timestamp,
            'checkpoint': str(self.checkpoint.get('epoch', 'N/A')),
            'best_dice': float(self.checkpoint.get('best_dice', 0)),
            'metrics': results[0],
            'std': results[1]
        }
        
        with open(output_dir / f'evaluation_summary_{timestamp}.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save per-sample results
        with open(output_dir / f'per_sample_results_{timestamp}.csv', 'w') as f:
            f.write('filename,dice,iou,precision,recall,f1,accuracy,hd95\n')
            for r in results[2]:
                f.write(f"{r['filename']},{r['dice']:.4f},{r['iou']:.4f},"
                        f"{r['precision']:.4f},{r['recall']:.4f},{r['f1']:.4f},{r['accuracy']:.4f},{r['hd95']:.4f}\n")
        
        return output_dir


def print_results(metrics, std):
    """Print evaluation results in formatted table"""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"{'Metric':<15} {'Mean':<15} {'Std':<15}")
    print("-"*70)
    for key in metrics.keys():
        print(f"{key.capitalize():<15} {metrics[key]:>10.4f} ± {std[key]:<10.4f}")
    print("="*70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate segmentation model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')
    parser.add_argument('--split', type=str, default='test', help='Dataset split')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--viz', action='store_true', help='Save visualizations')
    parser.add_argument('--threshold', action='store_true', help='Find optimal threshold')
    args = parser.parse_args()
    
    # Evaluate
    evaluator = Evaluator(args.checkpoint)
    results = evaluator.evaluate(
        args.dataset,
        split=args.split,
        save_viz=args.viz,
        viz_dir=f'{args.output}/visualizations',
        find_threshold=args.threshold
    )
    
    # Print and save
    print_results(results[0], results[1])
    output_dir = evaluator.save_results(results, args.output)
    
    # Save threshold search results if performed
    if results[3]:
        threshold_output = Path(args.output) / 'threshold_search.json'
        with open(threshold_output, 'w') as f:
            json.dump(results[3], f, indent=2)
        print(f"\nThreshold search results saved to: {threshold_output}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
