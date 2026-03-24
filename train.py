"""
Complete Training Pipeline for Medical Image Segmentation

Features:
- Multiple model support (UNet, Attention-UNet)
- Proper random seed setting for reproducibility
- TensorBoard logging
- Early stopping
- Learning rate warmup
- Gradient clipping
- Experiment tracking
"""

import os
import sys
import random
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from models import UNet, AttentionUNet, AttentionUNetLite
from models.losses import HybridDiceBCELoss, DiceLoss
from data.dataset import MedicalSegmentationDataset, get_train_transforms, get_val_transforms


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[INFO] Random seed set to {seed}")


def setup_logging(log_dir):
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'train_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file


def calculate_metrics(pred, target, threshold=0.5, smooth=1e-6):
    """Calculate segmentation metrics with proper handling for edge cases.
    
    Args:
        pred: Prediction tensor (B, 1, H, W) or (B, H, W)
        target: Ground truth tensor (B, 1, H, W) or (B, H, W)
        threshold: Threshold for binarization
        smooth: Smoothing factor for numerical stability
    
    Returns:
        Dictionary with metrics (all values in [0, 1] range)
    """
    # Ensure tensors are float
    pred = pred.float()
    target = target.float()
    
    # Handle different tensor shapes
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    
    # Binarize predictions
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    
    # Ensure target is binary
    target_binary = (target > 0.5).float()
    
    # Dice coefficient: 2 * |intersection| / (|pred| + |target|)
    intersection = (pred_binary * target_binary).sum()
    pred_sum = pred_binary.sum()
    target_sum = target_binary.sum()
    
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    # Clamp Dice to [0, 1] to handle numerical issues
    dice = torch.clamp(dice, 0.0, 1.0)
    
    # IoU (Jaccard): |intersection| / |union|
    union = pred_sum + target_sum - intersection
    iou = (intersection + smooth) / (union + smooth)
    iou = torch.clamp(iou, 0.0, 1.0)
    
    # Precision: TP / (TP + FP)
    tp = intersection
    fp = pred_sum - intersection
    fn = target_sum - intersection
    
    precision = (tp + smooth) / (tp + fp + smooth)
    precision = torch.clamp(precision, 0.0, 1.0)
    
    # Recall: TP / (TP + FN)
    recall = (tp + smooth) / (tp + fn + smooth)
    recall = torch.clamp(recall, 0.0, 1.0)
    
    # F1 Score
    f1 = 2 * precision * recall / (precision + recall + smooth)
    f1 = torch.clamp(f1, 0.0, 1.0)
    
    # Accuracy
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


class EarlyStopping:
    """Early stopping handler"""
    
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False


class Trainer:
    """Main trainer class"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup
        set_seed(config.get('seed', 42))
        self.log_dir = Path(config.get('log_dir', 'logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.writer = SummaryWriter(self.log_dir / 'tensorboard')
        self.experiment_name = f"{config['model']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logging.info(f"Experiment: {self.experiment_name}")
        logging.info(f"Device: {self.device}")
        
        # Model
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        logging.info(f"Model: {config['model']['name']}, Parameters: {self.model.get_params():,}")
        
        # Loss
        self.criterion = HybridDiceBCELoss(
            dice_weight=config['loss']['dice_weight'],
            bce_weight=config['loss']['bce_weight']
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs'],
            eta_min=float(config['training'].get('min_lr', 1e-6))
        )
        
        # Data
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training']['early_stopping_patience'],
            mode='max'
        )
        
        # Training history
        self.history = {
            'train_loss': [], 'train_dice': [], 'train_iou': [],
            'val_loss': [], 'val_dice': [], 'val_iou': []
        }
        
        self.best_dice = 0.0
        self.start_epoch = 0
        
    def _create_model(self):
        """Create model based on config"""
        model_name = self.config['model']['name']
        
        model_configs = {
            'unet': lambda: UNet(
                in_channels=self.config['data']['in_channels'],
                out_channels=self.config['data']['out_channels'],
                base_filters=self.config['model'].get('base_filters', 64)
            ),
            'attention_unet': lambda: AttentionUNet(
                in_channels=self.config['data']['in_channels'],
                out_channels=self.config['data']['out_channels'],
                base_filters=self.config['model'].get('base_filters', 64)
            ),
            'attention_unet_lite': lambda: AttentionUNetLite(
                in_channels=self.config['data']['in_channels'],
                out_channels=self.config['data']['out_channels'],
                base_filters=self.config['model'].get('base_filters', 32)
            ),
        }
        
        if model_name not in model_configs:
            raise ValueError(f"Unknown model: {model_name}")
            
        return model_configs[model_name]()
    
    def _create_dataloaders(self):
        """Create data loaders"""
        data_config = self.config['data']
        train_config = self.config['training']
        
        train_transform = get_train_transforms(
            tuple(data_config['image_size']),
            augment=data_config.get('augmentation', True)
        )
        val_transform = get_val_transforms(tuple(data_config['image_size']))
        
        full_dataset = MedicalSegmentationDataset(
            root_dir=data_config['root_dir'],
            split='train',
            transform=train_transform,
            image_size=tuple(data_config['image_size'])
        )
        
        # Split into train/val
        total_samples = len(full_dataset)
        val_size = int(total_samples * data_config.get('val_split', 0.2))
        train_size = total_samples - val_size
        
        indices = list(range(total_samples))
        random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(
            MedicalSegmentationDataset(
                root_dir=data_config['root_dir'],
                split='train',
                transform=val_transform,
                image_size=tuple(data_config['image_size'])
            ),
            val_indices
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=train_config.get('num_workers', 4),
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=train_config.get('num_workers', 4),
            pin_memory=True
        )
        
        logging.info(f"Dataset: {total_samples} samples (Train: {train_size}, Val: {val_size})")
        
        return train_loader, val_loader
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        metrics = {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Metrics
            with torch.no_grad():
                batch_metrics = calculate_metrics(outputs, masks)
            
            total_loss += loss.item()
            for k, v in batch_metrics.items():
                metrics[k] += v
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{batch_metrics["dice"]:.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_metrics = {k: v / len(self.train_loader) for k, v in metrics.items()}
        
        return avg_loss, avg_metrics
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        metrics = {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                total_loss += loss.item()
                
                batch_metrics = calculate_metrics(outputs, masks)
                for k, v in batch_metrics.items():
                    metrics[k] += v
        
        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = {k: v / len(self.val_loader) for k, v in metrics.items()}
        
        return avg_loss, avg_metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_dice': self.best_dice,
            'config': self.config,
            'history': self.history
        }
        
        # Regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'{self.experiment_name}_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / f'{self.experiment_name}_best.pth'
            torch.save(checkpoint, best_path)
            logging.info(f"Saved best model with Dice: {self.best_dice:.4f}")
        
        return checkpoint_path
    
    def train(self):
        """Main training loop"""
        epochs = self.config['training']['epochs']
        
        logging.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(self.start_epoch + 1, epochs + 1):
            logging.info(f"\n{'='*60}")
            logging.info(f"Epoch {epoch}/{epochs}")
            logging.info(f"{'='*60}")
            
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metrics = self.validate(epoch)
            
            # Step scheduler
            self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_dice'].append(train_metrics['dice'])
            self.history['train_iou'].append(train_metrics['iou'])
            self.history['val_loss'].append(val_loss)
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['val_iou'].append(val_metrics['iou'])
            
            # Logging
            logging.info(f"\nTrain Loss: {train_loss:.4f}")
            logging.info(f"Train - Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}, "
                        f"F1: {train_metrics['f1']:.4f}")
            logging.info(f"Val Loss: {val_loss:.4f}")
            logging.info(f"Val - Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}, "
                        f"F1: {val_metrics['f1']:.4f}")
            
            # TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Metrics/Dice_train', train_metrics['dice'], epoch)
            self.writer.add_scalar('Metrics/Dice_val', val_metrics['dice'], epoch)
            self.writer.add_scalar('Metrics/IoU_train', train_metrics['iou'], epoch)
            self.writer.add_scalar('Metrics/IoU_val', val_metrics['iou'], epoch)
            self.writer.add_scalar('LR', self.scheduler.get_last_lr()[0], epoch)
            
            # Save checkpoint
            is_best = val_metrics['dice'] > self.best_dice
            if is_best:
                self.best_dice = val_metrics['dice']
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.early_stopping(val_metrics['dice']):
                logging.info(f"\nEarly stopping triggered at epoch {epoch}")
                break
        
        logging.info(f"\nTraining completed! Best Dice: {self.best_dice:.4f}")
        self.writer.close()
        
        # Save training history
        import json
        history_path = self.log_dir / f'{self.experiment_name}_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logging.info(f"Training history saved to {history_path}")
        
        return self.best_dice


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train medical image segmentation model')
    parser.add_argument('--config', type=str, default='configs/train_isic.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default=None,
                        help='Override model name from config')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs from config')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config
    if args.model:
        config['model']['name'] = args.model
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path('logs') / config['model']['name'] / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'train.log'),
            logging.StreamHandler()
        ]
    )
    
    # Train
    trainer = Trainer(config)
    trainer.log_dir = log_dir
    trainer.writer = SummaryWriter(log_dir / 'tensorboard')
    
    best_dice = trainer.train()
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Dice Score: {best_dice:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
