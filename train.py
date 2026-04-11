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
import csv
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt

from models import UNet, AttentionUNet, AttentionUNetLite
from models.components.enhanced_attention_unet import EnhancedAttentionUNet
from models.losses import HybridDiceBCELoss, DiceLoss
from models.losses_enhanced import DeepSupervisionLoss, CombinedSegLoss
from data.dataset import MedicalSegmentationDataset, get_train_transforms, get_val_transforms


def set_seed(seed=42, deterministic=True):
    """Set random seed. Deterministic mode is optional for speed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    os.environ['PYTHONHASHSEED'] = str(seed)
    mode = "deterministic" if deterministic else "fast"
    print(f"[INFO] Random seed set to {seed} ({mode} mode)")


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
    
    hd95 = calculate_hd95(pred, target, threshold=threshold)

    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'accuracy': accuracy.item(),
        'hd95': hd95,
    }


def calculate_hd95(pred, target, threshold=0.5):
    """Approximate HD95 in pixels for 2D masks."""
    from scipy.ndimage import binary_erosion, distance_transform_edt

    try:
        pred = torch.sigmoid(pred).detach().float()
        target = target.detach().float()

        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)

        pred_b = (pred > threshold).cpu().numpy().astype(np.uint8).astype(bool)
        tgt_b = (target > 0.5).cpu().numpy().astype(np.uint8).astype(bool)

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
            d1 = d_g[p_s]
            d2 = d_p[g_s]
            all_d = np.concatenate([d1, d2]) if d1.size and d2.size else np.concatenate([d1, d2, np.array([0.0])])
            vals.append(float(np.percentile(all_d, 95)))

        finite = [v for v in vals if np.isfinite(v)]
        if not finite:
            return float('nan')
        return float(np.mean(finite))
    except Exception:
        return float('nan')


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
        self.use_amp = bool(config.get('training', {}).get('use_amp', True) and self.device.type == 'cuda')
        self.use_channels_last = bool(config.get('training', {}).get('channels_last', True) and self.device.type == 'cuda')
        self.hd95_train_every = int(config.get('training', {}).get('hd95_train_every', 0))
        self.hd95_val_every = int(config.get('training', {}).get('hd95_val_every', 1))
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Setup
        train_cfg = config.get('training', {})
        set_seed(config.get('seed', 42), deterministic=train_cfg.get('deterministic', False))
        if self.device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision('high')
            except Exception:
                pass
        log_cfg = config.get('logging', {})
        self.log_dir = Path(log_cfg.get('log_dir', config.get('log_dir', 'logs')))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(log_cfg.get('checkpoint_dir', config.get('checkpoint_dir', 'checkpoints')))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.writer = SummaryWriter(self.log_dir / 'tensorboard')
        self.experiment_name = f"{config['model']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logging.info(f"Experiment: {self.experiment_name}")
        logging.info(f"Device: {self.device}")
        
        # Model
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        if self.use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        logging.info(f"Model: {config['model']['name']}, Parameters: {self.model.get_params():,}")
        
        # Loss
        if config['loss']['name'] == 'DeepSupervisionLoss':
            self.criterion = DeepSupervisionLoss(
                dice_weight=config['loss']['dice_weight'],
                bce_weight=config['loss']['bce_weight']
            )
        elif config['loss']['name'] == 'CombinedSegLoss':
            self.criterion = CombinedSegLoss(
                dice_weight=config['loss']['dice_weight'],
                bce_weight=config['loss']['bce_weight'],
                lovasz_weight=config['loss'].get('lovasz_weight', 0.0)
            )
        else:
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
        sch = config['training'].get('scheduler', 'CosineAnnealingLR')
        if sch == 'CosineAnnealingWarmRestarts':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=int(config['training'].get('scheduler_T0', 10)),
                T_mult=int(config['training'].get('scheduler_T_mult', 1)),
                eta_min=float(config['training'].get('min_lr', 1e-6)),
            )
        else:
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
            'train_loss': [], 'train_dice': [], 'train_iou': [], 'train_hd95': [],
            'val_loss': [], 'val_dice': [], 'val_iou': [], 'val_hd95': []
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
            'enhanced_attention_unet': lambda: EnhancedAttentionUNet(
                in_channels=self.config['data']['in_channels'],
                out_channels=self.config['data']['out_channels'],
                base_filters=self.config['model'].get('base_filters', 64),
                deep_supervision=self.config['model'].get('deep_supervision', False)
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
        
        root_dir = Path(data_config['root_dir'])
        if (root_dir / 'train' / 'images').exists() and (root_dir / 'val' / 'images').exists():
            train_dataset = MedicalSegmentationDataset(
                root_dir=data_config['root_dir'],
                split='train',
                transform=train_transform,
                image_size=tuple(data_config['image_size'])
            )
            val_dataset = MedicalSegmentationDataset(
                root_dir=data_config['root_dir'],
                split='val',
                transform=val_transform,
                image_size=tuple(data_config['image_size'])
            )
            train_size = len(train_dataset)
            val_size = len(val_dataset)
            total_samples = train_size + val_size
        else:
            full_dataset = MedicalSegmentationDataset(
                root_dir=data_config['root_dir'],
                split='train',
                transform=train_transform,
                image_size=tuple(data_config['image_size'])
            )

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
            pin_memory=True,
            persistent_workers=bool(train_config.get('num_workers', 4) > 0),
            prefetch_factor=int(train_config.get('prefetch_factor', 2)) if train_config.get('num_workers', 4) > 0 else None,
            drop_last=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=train_config.get('num_workers', 4),
            pin_memory=True,
            persistent_workers=bool(train_config.get('num_workers', 4) > 0),
            prefetch_factor=int(train_config.get('prefetch_factor', 2)) if train_config.get('num_workers', 4) > 0 else None,
        )
        
        logging.info(f"Dataset: {total_samples} samples (Train: {train_size}, Val: {val_size})")
        
        return train_loader, val_loader
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        metrics = {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0, 'hd95': 0}
        counts = {k: 0 for k in metrics}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)
            if self.use_channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.config['training'].get('gradient_clip', 0) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Metrics - handle deep supervision (list of outputs)
            with torch.no_grad():
                if isinstance(outputs, list):
                    # Use main output for metrics
                    main_output = outputs[0]
                else:
                    main_output = outputs
                batch_metrics = calculate_metrics(main_output, masks)
                if self.hd95_train_every <= 0 or (batch_idx % self.hd95_train_every != 0):
                    batch_metrics['hd95'] = float('nan')
            
            total_loss += loss.item()
            for k, v in batch_metrics.items():
                if np.isfinite(v):
                    metrics[k] += v
                    counts[k] += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{batch_metrics["dice"]:.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_metrics = {
            k: (metrics[k] / counts[k] if counts[k] > 0 else float('nan'))
            for k in metrics
        }
        
        return avg_loss, avg_metrics
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        metrics = {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0, 'hd95': 0}
        counts = {k: 0 for k in metrics}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device, non_blocking=True)
                masks = batch['mask'].to(self.device, non_blocking=True)
                if self.use_channels_last:
                    images = images.contiguous(memory_format=torch.channels_last)

                with autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                total_loss += loss.item()
                
                # Handle deep supervision (list of outputs)
                if isinstance(outputs, list):
                    main_output = outputs[0]
                else:
                    main_output = outputs
                    
                batch_metrics = calculate_metrics(main_output, masks)
                if self.hd95_val_every > 0 and (epoch % self.hd95_val_every == 0):
                    pass
                else:
                    batch_metrics['hd95'] = float('nan')
                for k, v in batch_metrics.items():
                    if np.isfinite(v):
                        metrics[k] += v
                        counts[k] += 1
        
        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = {
            k: (metrics[k] / counts[k] if counts[k] > 0 else float('nan'))
            for k in metrics
        }
        
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

        # Keep only latest N epoch checkpoints to control disk usage
        keep_n = int(self.config.get('training', {}).get('keep_last_checkpoints', 2))
        if keep_n > 0:
            epoch_ckpts = sorted(
                self.checkpoint_dir.glob(f'{self.experiment_name}_epoch_*.pth'),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for old_ckpt in epoch_ckpts[keep_n:]:
                try:
                    old_ckpt.unlink()
                except Exception:
                    pass
        
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
            self.history['train_hd95'].append(train_metrics['hd95'])
            self.history['val_loss'].append(val_loss)
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['val_hd95'].append(val_metrics['hd95'])
            
            # Logging
            logging.info(f"\nTrain Loss: {train_loss:.4f}")
            logging.info(f"Train - Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}, "
                        f"F1: {train_metrics['f1']:.4f}, HD95: {train_metrics['hd95']:.3f}")
            logging.info(f"Val Loss: {val_loss:.4f}")
            logging.info(f"Val - Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}, "
                        f"F1: {val_metrics['f1']:.4f}, HD95: {val_metrics['hd95']:.3f}")
            
            # TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Metrics/Dice_train', train_metrics['dice'], epoch)
            self.writer.add_scalar('Metrics/Dice_val', val_metrics['dice'], epoch)
            self.writer.add_scalar('Metrics/IoU_train', train_metrics['iou'], epoch)
            self.writer.add_scalar('Metrics/IoU_val', val_metrics['iou'], epoch)
            self.writer.add_scalar('Metrics/HD95_train', train_metrics['hd95'], epoch)
            self.writer.add_scalar('Metrics/HD95_val', val_metrics['hd95'], epoch)
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

        # Save training history csv
        csv_path = self.log_dir / f'{self.experiment_name}_history.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_dice', 'train_iou', 'train_hd95', 'val_loss', 'val_dice', 'val_iou', 'val_hd95'])
            n = len(self.history['train_loss'])
            for i in range(n):
                writer.writerow([
                    i + 1,
                    self.history['train_loss'][i],
                    self.history['train_dice'][i],
                    self.history['train_iou'][i],
                    self.history['train_hd95'][i],
                    self.history['val_loss'][i],
                    self.history['val_dice'][i],
                    self.history['val_iou'][i],
                    self.history['val_hd95'][i],
                ])
        logging.info(f"Training history csv saved to {csv_path}")

        # Save simple curves plot
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        epochs_idx = list(range(1, len(self.history['train_loss']) + 1))
        axes[0].plot(epochs_idx, self.history['train_loss'], label='train_loss')
        axes[0].plot(epochs_idx, self.history['val_loss'], label='val_loss')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()
        axes[1].plot(epochs_idx, self.history['train_dice'], label='train_dice')
        axes[1].plot(epochs_idx, self.history['val_dice'], label='val_dice')
        axes[1].set_title('Dice')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[2].plot(epochs_idx, self.history['train_hd95'], label='train_hd95')
        axes[2].plot(epochs_idx, self.history['val_hd95'], label='val_hd95')
        axes[2].set_title('HD95 (px)')
        axes[2].set_xlabel('Epoch')
        axes[2].legend()
        fig.tight_layout()
        curve_path = self.log_dir / f'{self.experiment_name}_curves.png'
        fig.savefig(curve_path, dpi=150)
        plt.close(fig)
        logging.info(f"Training curves saved to {curve_path}")
        
        return self.best_dice


def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='Train medical image segmentation model')
    parser.add_argument('--config', type=str, default='configs/train_isic.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default=None,
                        help='Override model name from config')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs from config')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config
    if args.model:
        config['model']['name'] = args.model
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    target_epochs = config['training']['epochs']
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_log_dir = Path(config.get('logging', {}).get('log_dir', 'logs'))
    log_dir = base_log_dir / config['model']['name'] / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(config.get('logging', {}).get('checkpoint_dir', 'checkpoints')) / config['model']['name'] / timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'train.log'),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Train
        trainer = Trainer(config)
        trainer.log_dir = log_dir
        trainer.checkpoint_dir = checkpoint_dir
        trainer.writer = SummaryWriter(log_dir / 'tensorboard')
        
        # Resume from checkpoint if specified
        if args.resume:
            print(f"Resuming from: {args.resume}")
            logging.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            trainer.start_epoch = checkpoint['epoch']
            trainer.best_dice = checkpoint.get('best_dice', 0)
            if 'history' in checkpoint:
                trainer.history = checkpoint['history']
            print(f"Resumed from epoch {trainer.start_epoch}, best_dice: {trainer.best_dice:.4f}")
            logging.info(f"Resumed from epoch {trainer.start_epoch}, best_dice: {trainer.best_dice:.4f}")
        
        best_dice = trainer.train()
        
        # Check if training completed all epochs
        completed_epochs = len(trainer.history['train_loss'])
        if completed_epochs < target_epochs - trainer.start_epoch:
            logging.error(f"Training incomplete: {completed_epochs}/{target_epochs - trainer.start_epoch} epochs")
            print(f"\nTraining incomplete: {completed_epochs} epochs completed")
            sys.exit(1)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Dice Score: {best_dice:.4f}")
        print(f"{'='*60}")
        
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
