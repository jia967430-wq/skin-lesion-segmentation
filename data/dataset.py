"""
Medical Image Segmentation Dataset Loader

Supports:
- ISIC 2018 Skin Lesion Segmentation
- Custom datasets with similar structure
- Various augmentation strategies
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size=(256, 256), augment=True):
    """
    Get training augmentation transforms
    
    Args:
        image_size: Target image size (H, W)
        augment: Whether to apply augmentations
    """
    if not augment:
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,
            p=0.5
        ),
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.05), p=1),
            A.GaussianBlur(blur_limit=(3, 5), p=1),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transforms(image_size=(256, 256)):
    """Get validation transforms (no augmentation)"""
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_test_transforms(image_size=(256, 256)):
    """Get test transforms (same as validation)"""
    return get_val_transforms(image_size)


class MedicalSegmentationDataset(Dataset):
    """
    Medical Image Segmentation Dataset
    
    Expected directory structure:
    root_dir/
    ├── train/
    │   ├── images/  (jpg, png, etc.)
    │   └── masks/   (png, single channel)
    ├── val/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/
    
    OR single split:
    root_dir/
    ├── images/
    └── masks/
    """
    
    def __init__(self, root_dir, split='train', transform=None, image_size=(256, 256)):
        """
        Args:
            root_dir: Root directory of the dataset
            split: 'train', 'val', 'test', or 'all' for combined
            transform: Albumentations transforms
            image_size: Target image size (H, W)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        # Check for different directory structures
        if (self.root_dir / 'train' / 'images').exists():
            # Multi-split structure
            self.image_dir = self.root_dir / split / 'images'
            self.mask_dir = self.root_dir / split / 'masks'
        else:
            # Single split structure
            self.image_dir = self.root_dir / 'images'
            self.mask_dir = self.root_dir / 'masks'
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        # Get image files
        self.image_files = self._get_image_files()
        
        print(f"[INFO] Loaded {len(self.image_files)} images from {split} split")
    
    def _get_image_files(self):
        """Get list of image files"""
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        
        files = []
        for f in os.listdir(self.image_dir):
            ext = os.path.splitext(f.lower())[1]
            if ext in supported_formats:
                files.append(f)
        
        return sorted(files)
    
    def _get_mask_path(self, image_name):
        """Get corresponding mask path for an image"""
        # Common mask naming conventions
        base_name = os.path.splitext(image_name)[0]
        mask_name = base_name + '.png'  # Assume mask is always PNG
        
        # Try different possible locations
        possible_masks = [
            self.mask_dir / mask_name,
            self.mask_dir / base_name.replace('ISIC_', '') / mask_name,
            self.mask_dir / image_name.replace('.jpg', '_segmentation.png').replace('.jpeg', '_segmentation.png')
        ]
        
        for mask_path in possible_masks:
            if mask_path.exists():
                return mask_path
        
        # If no exact match, try to find by base name
        for f in os.listdir(self.mask_dir):
            if base_name in f:
                return self.mask_dir / f
        
        raise FileNotFoundError(f"Mask not found for {image_name}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = self.image_dir / img_name
        mask_path = self._get_mask_path(img_name)
        
        # Load image
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Load mask
        mask = np.array(Image.open(mask_path).convert('L'))
        
        # Binarize mask if needed
        if mask.max() > 1:
            mask = (mask > 127).astype(np.float32)
        else:
            mask = mask.astype(np.float32)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Default preprocessing
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return {
            'image': image,
            'mask': mask,
            'filename': img_name
        }


class ISICDataset(MedicalSegmentationDataset):
    """ISIC 2018 Dataset wrapper"""
    
    def _get_mask_path(self, image_name):
        """ISIC specific mask path resolution"""
        base_name = os.path.splitext(image_name)[0]
        # ISIC masks are named with _segmentation suffix
        mask_name = base_name + '_segmentation.png'
        mask_path = self.mask_dir / mask_name
        
        if mask_path.exists():
            return mask_path
        
        # Try without _segmentation
        mask_path = self.mask_dir / (base_name + '.png')
        if mask_path.exists():
            return mask_path
        
        raise FileNotFoundError(f"ISIC mask not found for {image_name}")


class PairedDataset(Dataset):
    """Dataset for custom paired image-mask data"""
    
    def __init__(self, image_dir, mask_dir, transform=None, image_size=(256, 256)):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_size = image_size
        
        # Get paired files
        self.image_files = sorted([
            f for f in os.listdir(image_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        # Verify pairs exist
        self.valid_indices = []
        for i, img_file in enumerate(self.image_files):
            mask_file = self._get_mask_name(img_file)
            if (self.mask_dir / mask_file).exists():
                self.valid_indices.append(i)
        
        print(f"[INFO] Loaded {len(self.valid_indices)} valid image-mask pairs")
    
    def _get_mask_name(self, image_name):
        """Convert image name to mask name"""
        base = os.path.splitext(image_name)[0]
        return base + '.png'
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        img_name = self.image_files[real_idx]
        
        image = np.array(Image.open(self.image_dir / img_name).convert('RGB'))
        mask = np.array(Image.open(self.mask_dir / self._get_mask_name(img_name)).convert('L'))
        
        if mask.max() > 1:
            mask = (mask > 127).astype(np.float32)
        else:
            mask = mask.astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return {'image': image, 'mask': mask, 'filename': img_name}


# Utility for downloading ISIC dataset
def download_isic2018(output_dir, api_key=None):
    """
    Download ISIC 2018 dataset
    
    Note: Requires registration at https://isic-archive.com/
    """
    print("[INFO] ISIC 2018 download requires:")
    print("1. Register at https://isic-archive.com/")
    print("2. Download via the web interface or API")
    print(f"3. Extract to: {output_dir}")
    
    # Create directory structure
    output_dir = Path(output_dir)
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'masks').mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Created directory structure at {output_dir}")


from pathlib import Path
