"""
ISIC 2018 Dataset Download Script

Downloads the ISIC 2018 skin lesion segmentation dataset from official sources.

Options:
1. Manual download (recommended for large datasets)
2. Kaggle API (if you have a Kaggle account)
3. Direct download links (limited)

Usage:
    python scripts/download_isic.py --method manual
    python scripts/download_isic.py --method kaggle --username your_username --key your_api_key
"""

import os
import sys
import argparse
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm
import urllib.request


def download_file(url: str, output_path: Path, desc: str = "Downloading"):
    """Download file with progress bar"""
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
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
        print(f"[ERROR] Failed to download: {e}")
        return False


def download_from_kaggle(username: str, key: str, dataset_name: str, output_dir: Path):
    """Download dataset using Kaggle API"""
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Authenticate
        os.environ['KAGGLE_USERNAME'] = username
        os.environ['KAGGLE_KEY'] = key
        
        api = KaggleApi()
        api.authenticate()
        
        print(f"[INFO] Downloading {dataset_name} from Kaggle...")
        
        # Download dataset
        api.dataset_download_files(dataset_name, path=output_dir, unzip=True)
        
        print(f"[INFO] Download complete!")
        return True
    except ImportError:
        print("[ERROR] Kaggle library not installed. Run: pip install kaggle")
        return False
    except Exception as e:
        print(f"[ERROR] Kaggle download failed: {e}")
        return False


def prepare_isic_structure(source_dir: Path, output_dir: Path):
    """Prepare ISIC dataset in the required directory structure"""
    print("[INFO] Preparing dataset structure...")
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'masks').mkdir(parents=True, exist_ok=True)
    
    # Common image extensions
    image_extensions = ['.jpg', '.jpeg', '.png']
    
    # Find all images in source directory
    source_images = []
    for ext in image_extensions:
        source_images.extend(source_dir.glob(f'*{ext}'))
    
    print(f"[INFO] Found {len(source_images)} images in source directory")
    
    # Split into train/val/test (80/10/10)
    import random
    random.shuffle(source_images)
    
    n_total = len(source_images)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    splits = {
        'train': source_images[:n_train],
        'val': source_images[n_train:n_train + n_val],
        'test': source_images[n_train + n_val:]
    }
    
    # Copy files to organized structure
    for split_name, images in splits.items():
        for img_path in tqdm(images, desc=f'Copying {split_name}'):
            # Find corresponding mask (assumes _segmentation suffix or same name with .png)
            base_name = img_path.stem
            
            # Look for mask in same directory
            mask_candidates = [
                img_path.parent / f'{base_name}_segmentation.png',
                img_path.parent / f'{base_name}.png',
            ]
            
            mask_path = None
            for candidate in mask_candidates:
                if candidate.exists():
                    mask_path = candidate
                    break
            
            # Copy image
            dest_img = output_dir / split_name / 'images' / img_path.name
            shutil.copy2(img_path, dest_img)
            
            # Copy mask if found
            if mask_path:
                dest_mask = output_dir / split_name / 'masks' / mask_path.name
                shutil.copy2(mask_path, dest_mask)
    
    print(f"[INFO] Dataset prepared at: {output_dir}")
    
    # Print statistics
    for split in ['train', 'val', 'test']:
        n_images = len(list((output_dir / split / 'images').glob('*')))
        n_masks = len(list((output_dir / split / 'masks').glob('*')))
        print(f"  {split.capitalize()}: {n_images} images, {n_masks} masks")


def create_sample_predictions(model_path: str, dataset_path: str, output_dir: str, num_samples: int = 4):
    """Generate sample prediction visualizations using a trained model"""
    print("[INFO] Generating sample predictions...")
    
    try:
        import torch
        from pathlib import Path
        from data.dataset import MedicalSegmentationDataset, get_test_transforms
        from utils.visualization import visualize_prediction, create_overlay_visualization
        from models import UNet
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint.get('config', {})
        
        model = UNet(in_channels=3, out_channels=1, base_filters=64)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Load dataset
        dataset = MedicalSegmentationDataset(
            root_dir=dataset_path,
            split='test',
            transform=get_test_transforms((224, 224)),
            image_size=(224, 224)
        )
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with torch.no_grad():
            for i in range(min(num_samples, len(dataset))):
                sample = dataset[i]
                image = sample['image'].unsqueeze(0).to(device)
                mask = sample['mask']
                
                pred = model(image)
                
                # Visualize
                fig = visualize_prediction(
                    image.squeeze().cpu().numpy(),
                    mask.numpy(),
                    pred.squeeze().cpu().numpy(),
                    save_path=str(output_path / f'sample_{i+1}.png')
                )
                plt.close(fig)
                
                # Overlay
                fig2 = create_overlay_visualization(
                    image.squeeze().cpu().numpy(),
                    mask.numpy(),
                    pred.squeeze().cpu().numpy(),
                    save_path=str(output_path / f'overlay_{i+1}.png')
                )
                plt.close(fig2)
        
        print(f"[INFO] Sample predictions saved to: {output_path}")
        
    except Exception as e:
        print(f"[WARNING] Could not generate sample predictions: {e}")


def print_instructions():
    """Print detailed download instructions"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        ISIC 2018 Dataset Download Instructions                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  OPTION 1: Manual Download (Recommended)                                      ║
║  ─────────────────────────────────────                                       ║
║  1. Visit: https://challenge.isic-archive.com/data/2018                     ║
║  2. Register for a free account                                              ║
║  3. Download the following files:                                            ║
║     - ISIC 2018: Training Data (images)                                     ║
║     - ISIC 2018: Ground Truth (segmentation masks)                          ║
║  4. Extract both archives to the same folder                                 ║
║  5. Run: python scripts/download_isic.py --method manual --source /path     ║
║                                                                              ║
║  OPTION 2: Kaggle API                                                       ║
║  ─────────────────────────                                                   ║
║  1. Get your Kaggle credentials:                                             ║
║     https://www.kaggle.com/account → Create API Token                       ║
║  2. Install kaggle: pip install kaggle                                     ║
║  3. Run: python scripts/download_isic.py \\                                  ║
║           --method kaggle \\                                                  ║
║           --username your_username \\                                         ║
║           --key your_api_key                                                 ║
║                                                                              ║
║  OPTION 3: Use Synthetic Data (for testing)                                  ║
║  ─────────────────────────────────────────                                  ║
║  Run: python scripts/prepare_data.py --root data/ISIC2018 --synthetic        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(
        description='Download and prepare ISIC 2018 dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show download instructions
  python scripts/download_isic.py --help
  
  # Manual preparation (after downloading)
  python scripts/download_isic.py --method manual --source /path/to/extracted/data
  
  # Kaggle download
  python scripts/download_isic.py --method kaggle --username user --key api_key
  
  # Generate synthetic data
  python scripts/prepare_data.py --root data/ISIC2018 --synthetic
        """
    )
    
    parser.add_argument('--method', type=str, choices=['manual', 'kaggle', 'synthetic', 'info'],
                       default='info', help='Download method')
    parser.add_argument('--source', type=str, help='Source directory for manual method')
    parser.add_argument('--output', type=str, default='data/ISIC2018',
                       help='Output directory for dataset')
    parser.add_argument('--username', type=str, help='Kaggle username')
    parser.add_argument('--key', type=str, help='Kaggle API key')
    parser.add_argument('--dataset', type=str, default='isic2018',
                       help='Kaggle dataset name')
    
    args = parser.parse_args()
    
    if args.method == 'info':
        print_instructions()
        return
    
    output_dir = Path(args.output)
    
    if args.method == 'manual':
        if not args.source:
            print("[ERROR] --source required for manual method")
            return
        
        source_dir = Path(args.source)
        if not source_dir.exists():
            print(f"[ERROR] Source directory not found: {source_dir}")
            return
        
        prepare_isic_structure(source_dir, output_dir)
    
    elif args.method == 'kaggle':
        if not args.username or not args.key:
            print("[ERROR] --username and --key required for Kaggle method")
            return
        
        download_dir = output_dir / 'temp'
        download_dir.mkdir(parents=True, exist_ok=True)
        
        success = download_from_kaggle(args.username, args.key, args.dataset, download_dir)
        
        if success:
            prepare_isic_structure(download_dir, output_dir)
            shutil.rmtree(download_dir)
    
    elif args.method == 'synthetic':
        print("[INFO] Use scripts/prepare_data.py to generate synthetic data")
        print("[INFO] Command: python scripts/prepare_data.py --root data/ISIC2018 --synthetic")


if __name__ == '__main__':
    main()
