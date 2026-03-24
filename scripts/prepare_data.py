#!/usr/bin/env python3
"""
ISIC 2018 Dataset Download and Preparation Script

This script downloads and prepares the ISIC 2018 skin lesion segmentation dataset.

Note: ISIC 2018 requires registration. This script provides:
1. Instructions for manual download
2. Automated structure preparation
3. Synthetic data generation for testing

Dataset URL: https://challenge.isic-archive.com/data/2018
"""

import os
import sys
import argparse
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw


def download_file(url, output_path, desc="Downloading"):
    """Download file with progress bar"""
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=desc,
                mininterval=1.0
            ) as pbar:
                for chunk in iter(lambda: response.read(8192), b''):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def prepare_directory_structure(root_dir):
    """Create standard directory structure"""
    root = Path(root_dir)
    
    for split in ['train', 'val', 'test']:
        (root / split / 'images').mkdir(parents=True, exist_ok=True)
        (root / split / 'masks').mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Created directory structure at {root}")
    return root


def generate_synthetic_isic(root_dir, num_train=100, num_val=30, num_test=30):
    """
    Generate synthetic skin lesion data with realistic characteristics.
    
    Creates data that mimics ISIC 2018 characteristics:
    - Dermoscopic images with varying skin tones
    - Circular/oval lesions with irregular boundaries
    - Various lesion sizes and positions
    """
    root = Path(root_dir)
    
    print(f"[INFO] Generating synthetic ISIC-style dataset...")
    
    splits = {
        'train': num_train,
        'val': num_val,
        'test': num_test
    }
    
    for split, n_samples in splits.items():
        print(f"[INFO] Generating {n_samples} {split} samples...")
        
        for i in tqdm(range(n_samples), desc=f'{split.capitalize()}'):
            # Image size similar to ISIC (standardized to 224x224)
            img_size = 224
            
            # Background skin colors (varying tones)
            skin_colors = [
                (241, 194, 150),  # Light
                (224, 172, 116),  # Medium-light
                (198, 134, 66),   # Medium
                (141, 85, 36),    # Medium-dark
                (94, 58, 28),     # Dark
            ]
            
            # Create base image with skin color
            bg_color = skin_colors[np.random.randint(0, len(skin_colors))]
            image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            
            for c in range(3):
                noise = np.random.randint(-15, 15, (img_size, img_size))
                image[:, :, c] = np.clip(bg_color[c] + noise, 0, 255)
            
            # Add texture (dermoscopic patterns)
            noise = np.random.normal(0, 3, (img_size, img_size, 3))
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
            
            # Create lesion mask
            mask = np.zeros((img_size, img_size), dtype=np.uint8)
            
            # Lesion parameters
            center_x = np.random.randint(img_size // 4, 3 * img_size // 4)
            center_y = np.random.randint(img_size // 4, 3 * img_size // 4)
            base_radius = np.random.randint(25, 60)
            
            # Create irregular lesion shape
            y, x = np.ogrid[:img_size, :img_size]
            
            # Add some irregularity to the boundary
            angles = np.linspace(0, 2 * np.pi, 36)
            irregular_radii = base_radius + np.random.randint(-10, 10, len(angles))
            
            for j, (angle, r) in enumerate(zip(angles, irregular_radii)):
                px = int(center_x + r * np.cos(angle))
                py = int(center_y + r * np.sin(angle))
                
                if 0 <= px < img_size and 0 <= py < img_size:
                    rr = np.random.randint(5, 10)
                    yy, xx = np.ogrid[:img_size, :img_size]
                    mask[np.sqrt((xx - px)**2 + (yy - py)**2) < rr] = 1
            
            # Ensure main lesion area is covered
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            main_mask = dist < base_radius
            mask[main_mask] = 1
            
            # Add some protrusions
            num_protrusions = np.random.randint(2, 5)
            for _ in range(num_protrusions):
                px = center_x + np.random.randint(-base_radius, base_radius)
                py = center_y + np.random.randint(-base_radius, base_radius)
                pr = np.random.randint(8, 15)
                yy, xx = np.ogrid[:img_size, :img_size]
                mask[np.sqrt((xx - px)**2 + (yy - py)**2) < pr] = 1
            
            # Lesion color (slightly different from skin)
            lesion_colors = [
                (180, 120, 80),   # Brown
                (150, 80, 60),    # Dark brown
                (200, 150, 120),  # Light brown
                (120, 70, 50),    # Very dark
                (160, 100, 70),   # Medium brown
            ]
            lesion_color = lesion_colors[np.random.randint(0, len(lesion_colors))]
            
            # Apply lesion color with some variation
            for c in range(3):
                lesion_channel = np.full((img_size, img_size), lesion_color[c], dtype=np.uint8)
                lesion_noise = np.random.randint(-20, 20, (img_size, img_size))
                lesion_channel = np.clip(lesion_channel + lesion_noise, 0, 255)
                image[:, :, c] = np.where(mask == 1, lesion_channel, image[:, :, c])
            
            # Add some fine details (hair simulation)
            num_hairs = np.random.randint(0, 8)
            for _ in range(num_hairs):
                h_start = np.random.randint(0, img_size)
                h_length = np.random.randint(20, 60)
                h_angle = np.random.uniform(0, 2 * np.pi)
                
                for l in range(h_length):
                    px = int(h_start + l * np.cos(h_angle))
                    py = int(h_start + l * np.sin(h_angle))
                    if 0 <= px < img_size and 0 <= py < img_size:
                        image[py, px] = np.clip(image[py, px] - 40, 0, 255)
            
            # Save image
            img_pil = Image.fromarray(image)
            img_path = root / split / 'images' / f'ISIC_{split}_{i:05d}.jpg'
            img_pil.save(img_path, quality=95)
            
            # Save mask
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_path = root / split / 'masks' / f'ISIC_{split}_{i:05d}_segmentation.png'
            mask_pil.save(mask_path)
    
    print(f"[INFO] Synthetic dataset generation complete!")
    print(f"[INFO] Total samples: {sum(splits.values())}")
    print(f"       Train: {splits['train']}")
    print(f"       Val: {splits['val']}")
    print(f"       Test: {splits['test']}")


def verify_dataset(root_dir):
    """Verify dataset structure and return statistics"""
    root = Path(root_dir)
    
    stats = {}
    
    for split in ['train', 'val', 'test']:
        img_dir = root / split / 'images'
        mask_dir = root / split / 'masks'
        
        if not img_dir.exists() or not mask_dir.exists():
            stats[split] = {'images': 0, 'masks': 0}
            continue
        
        n_images = len(list(img_dir.glob('*')))
        n_masks = len(list(mask_dir.glob('*')))
        
        stats[split] = {'images': n_images, 'masks': n_masks}
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Prepare ISIC 2018 dataset')
    parser.add_argument('--root', type=str, default='data/ISIC2018',
                        help='Root directory for dataset')
    parser.add_argument('--synthetic', action='store_true',
                        help='Generate synthetic data for testing')
    parser.add_argument('--train', type=int, default=100,
                        help='Number of synthetic train samples')
    parser.add_argument('--val', type=int, default=30,
                        help='Number of synthetic validation samples')
    parser.add_argument('--test', type=int, default=30,
                        help='Number of synthetic test samples')
    parser.add_argument('--verify', action='store_true',
                        help='Verify existing dataset')
    args = parser.parse_args()
    
    if args.verify:
        stats = verify_dataset(args.root)
        print("\nDataset Statistics:")
        print("-" * 40)
        for split, counts in stats.items():
            print(f"{split.capitalize()}: {counts['images']} images, {counts['masks']} masks")
        return
    
    # Prepare directory structure
    root = prepare_directory_structure(args.root)
    
    if args.synthetic:
        generate_synthetic_isic(
            args.root,
            num_train=args.train,
            num_val=args.val,
            num_test=args.test
        )
        
        # Verify
        stats = verify_dataset(args.root)
        print("\nFinal Dataset Statistics:")
        print("-" * 40)
        for split, counts in stats.items():
            print(f"{split.capitalize()}: {counts['images']} images, {counts['masks']} masks")
    else:
        print("\n" + "="*60)
        print("ISIC 2018 Dataset Preparation")
        print("="*60)
        print("""
To use the real ISIC 2018 dataset:

1. Register at: https://isic-archive.com/
2. Download the dataset from:
   - Training data: https://challenge.isic-archive.com/data/2018
   - Ground truth: Same page under "Segmentation"

3. Extract the downloaded archives

4. Organize the files as:
   data/ISIC2018/
   ├── train/
   │   ├── images/   <- ISIC training images
   │   └── masks/    <- Ground truth segmentations
   ├── val/
   │   ├── images/
   │   └── masks/
   └── test/
       ├── images/
       └── masks/

Alternatively, use --synthetic flag to generate test data:
   python scripts/prepare_data.py --root data/ISIC2018 --synthetic --train 100 --val 30 --test 30
""")


if __name__ == '__main__':
    main()
