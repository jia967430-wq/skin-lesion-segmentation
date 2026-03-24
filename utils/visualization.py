"""
Visualization Utilities for Medical Image Segmentation

This module provides comprehensive visualization tools for:
- Segmentation results comparison
- Training curves plotting
- Model architecture visualization
- Feature map visualization
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from pathlib import Path
import seaborn as sns
from typing import List, Tuple, Optional, Union


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def denormalize_image(image: np.ndarray, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) -> np.ndarray:
    """Denormalize an image from ImageNet normalization.
    
    Args:
        image: Normalized image array (H, W, C) or (C, H, W)
        mean: Mean values for normalization
        std: Std values for normalization
    
    Returns:
        Denormalized image in [0, 1] range
    """
    if image.shape[0] == 3:  # (C, H, W) format
        image = image.transpose(1, 2, 0)
    
    mean = np.array(mean)
    std = np.array(std)
    
    denorm = image * std + mean
    denorm = np.clip(denorm, 0, 1)
    
    return denorm


def create_segmentation_colormap():
    """Create a colormap for segmentation visualization.
    
    Returns:
        ListedColormap with distinct colors for different classes
    """
    colors = [
        '#000000',  # Background - Black
        '#FF0000',  # Class 1 - Red
        '#00FF00',  # Class 2 - Green
        '#0000FF',  # Class 3 - Blue
        '#FFFF00',  # Class 4 - Yellow
        '#FF00FF',  # Class 5 - Magenta
        '#00FFFF',  # Class 6 - Cyan
    ]
    return mcolors.ListedColormap(colors)


def visualize_prediction(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    probability: Optional[np.ndarray] = None,
    title: str = "",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 4),
    show_metrics: bool = True,
    dice: Optional[float] = None,
    iou: Optional[float] = None
) -> plt.Figure:
    """Create a comprehensive visualization of segmentation results.
    
    Args:
        image: Input image (H, W, 3) or normalized (C, H, W)
        ground_truth: Ground truth mask (H, W) binary or multi-class
        prediction: Predicted mask (H, W) binary or multi-class
        probability: Prediction probability map (H, W) for binary
        title: Title for the figure
        save_path: Path to save the figure
        figsize: Figure size
        show_metrics: Whether to show metrics in title
        dice: Dice score to display
        iou: IoU score to display
    
    Returns:
        matplotlib Figure object
    """
    # Handle tensor inputs
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    if probability is not None and isinstance(probability, torch.Tensor):
        probability = probability.cpu().numpy()
    
    # Denormalize image if needed
    if image.max() <= 1.0 and image.min() < 0:
        image = denormalize_image(image)
    
    # Handle different shapes
    if ground_truth.ndim == 3:
        ground_truth = ground_truth.squeeze()
    if prediction.ndim == 3:
        prediction = prediction.squeeze()
    
    # Create figure
    n_cols = 3 if probability is None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]
    
    # 1. Input Image
    axes[0].imshow(image)
    axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Ground Truth
    if ground_truth.max() > 1:
        ground_truth = (ground_truth > 0.5).astype(np.float32)
    
    axes[1].imshow(ground_truth, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Prediction
    if prediction.max() > 1:
        prediction = (prediction > 0.5).astype(np.float32)
    
    axes[2].imshow(prediction, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Prediction', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # 4. Probability Map (optional)
    if probability is not None:
        im = axes[3].imshow(probability, cmap='jet', vmin=0, vmax=1)
        axes[3].set_title('Probability Map', fontsize=12, fontweight='bold')
        axes[3].axis('off')
        plt.colorbar(im, ax=axes[3], fraction=0.046)
    
    # Add title with metrics
    if show_metrics and dice is not None:
        title_text = f'{title}\nDice: {dice:.4f}'
        if iou is not None:
            title_text += f', IoU: {iou:.4f}'
        fig.suptitle(title_text, fontsize=14, fontweight='bold')
    elif title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"[INFO] Saved visualization to: {save_path}")
    
    return fig


def create_overlay_visualization(
    image: np.ndarray,
    mask: np.ndarray,
    prediction: np.ndarray,
    save_path: Optional[str] = None,
    alpha: float = 0.5
) -> plt.Figure:
    """Create overlay visualization showing prediction on image.
    
    Args:
        image: Input image
        mask: Ground truth mask
        prediction: Predicted mask
        save_path: Path to save figure
        alpha: Overlay transparency
    
    Returns:
        matplotlib Figure object
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    
    # Denormalize image if needed
    if image.max() <= 1.0 and image.min() < 0:
        image = denormalize_image(image)
    
    # Binarize masks
    if mask.max() > 1:
        mask = (mask > 0.5).astype(np.float32)
    if prediction.max() > 1:
        prediction = (prediction > 0.5).astype(np.float32)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Image only
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # Ground truth overlay
    overlay_gt = image.copy()
    mask_bool = mask > 0.5
    overlay_gt[mask_bool] = np.clip(overlay_gt[mask_bool] * 0.5 + np.array([0, 1, 0]) * 0.5, 0, 1)
    axes[1].imshow(overlay_gt)
    axes[1].set_title('Ground Truth Overlay', fontsize=12)
    axes[1].axis('off')
    
    # Prediction overlay
    overlay_pred = image.copy()
    correct = (prediction == mask)
    pred_bool = prediction > 0.5
    
    tp = pred_bool & mask_bool
    fp = pred_bool & ~mask_bool
    fn = ~pred_bool & mask_bool
    
    overlay_pred[tp] = np.clip(overlay_pred[tp] * 0.7 + np.array([0, 1, 0]) * 0.3, 0, 1)  # Green - TP
    overlay_pred[fp] = np.clip(overlay_pred[fp] * 0.7 + np.array([1, 0, 0]) * 0.3, 0, 1)  # Red - FP
    overlay_pred[fn] = np.clip(overlay_pred[fn] * 0.7 + np.array([0, 0, 1]) * 0.3, 0, 1)  # Blue - FN
    
    axes[2].imshow(overlay_pred)
    axes[2].set_title('Prediction Overlay\n(Green=TP, Red=FP, Blue=FN)', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_training_curves(
    history: dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """Plot training and validation curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_dice', 'val_dice', etc.
        save_path: Path to save the figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Loss curves
    if 'train_loss' in history:
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Dice curves
    if 'train_dice' in history:
        axes[1].plot(epochs, history['train_dice'], 'b-', label='Train Dice', linewidth=2)
    if 'val_dice' in history:
        axes[1].plot(epochs, history['val_dice'], 'r-', label='Val Dice', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Dice Score', fontsize=12)
    axes[1].set_title('Training and Validation Dice', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"[INFO] Saved training curves to: {save_path}")
    
    return fig


def plot_metric_comparison(
    results: dict,
    metric: str = 'dice',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot comparison of metrics across different models.
    
    Args:
        results: Dictionary mapping model names to their metrics
        metric: Metric to compare ('dice', 'iou', 'precision', etc.)
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    models = list(results.keys())
    values = [results[m].get(metric, 0) for m in models]
    std = [results[m].get(f'{metric}_std', 0) for m in models]
    
    # Sort by value
    sorted_pairs = sorted(zip(models, values, std), key=lambda x: x[1], reverse=True)
    models, values, std = zip(*sorted_pairs)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = sns.color_palette("husl", len(models))
    bars = ax.bar(models, values, yerr=std, capsize=5, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(f'{metric.upper()} Score', fontsize=12)
    ax.set_title(f'Model Comparison - {metric.upper()}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"[INFO] Saved comparison plot to: {save_path}")
    
    return fig


def create_confusion_matrix_visual(
    tp: int, fp: int, fn: int, tn: int,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create confusion matrix visualization for segmentation.
    
    Args:
        tp: True Positives
        fp: False Positives
        fn: False Negatives
        tn: True Negatives
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure object
    """
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion matrix
    cm = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'])
    axes[0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Ground Truth')
    axes[0].set_xlabel('Prediction')
    
    # Metrics summary
    metrics_text = f"""
    Segmentation Metrics Summary
    ============================
    
    Precision: {precision:.4f}
    Recall:    {recall:.4f}
    F1 Score:  {f1:.4f}
    
    Confusion Matrix Values:
    - True Positives:  {tp:,}
    - False Positives: {fp:,}
    - False Negatives: {fn:,}
    - True Negatives:  {tn:,}
    """
    
    axes[1].text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                  verticalalignment='center', transform=axes[1].transAxes)
    axes[1].axis('off')
    axes[1].set_title('Metrics Summary', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig


def visualize_batch_predictions(
    images: torch.Tensor,
    masks: torch.Tensor,
    predictions: torch.Tensor,
    num_samples: int = 4,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Visualize predictions for a batch of images.
    
    Args:
        images: Batch of images (B, C, H, W)
        masks: Batch of masks (B, H, W) or (B, 1, H, W)
        predictions: Batch of predictions (B, H, W) or (B, 1, H, W)
        num_samples: Number of samples to visualize
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure object
    """
    batch_size = min(num_samples, images.shape[0])
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        img = images[i].cpu().numpy()
        mask = masks[i].cpu().numpy()
        pred = predictions[i].cpu().numpy()
        
        # Squeeze extra dimensions
        if mask.ndim == 3:
            mask = mask.squeeze()
        if pred.ndim == 3:
            pred = pred.squeeze()
        
        # Denormalize image
        if img.max() <= 1.0:
            img = denormalize_image(img)
        
        # Row: Image, Ground Truth, Prediction
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Image {i+1}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Ground Truth {i+1}')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Prediction {i+1}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig
