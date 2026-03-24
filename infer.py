"""
Inference Script for Medical Image Segmentation

Performs inference on single images or batches of images.
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import UNet, AttentionUNet, AttentionUNetLite
from data.dataset import get_test_transforms


class InferenceEngine:
    """Inference engine for segmentation models"""
    
    def __init__(self, checkpoint_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = self.checkpoint.get('config', {})
        
        # Create model
        self.model = self._create_model()
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.image_size = tuple(self.config.get('data', {}).get('image_size', [224, 224]))
        
        print(f"[INFO] Model loaded: {checkpoint_path}")
        print(f"[INFO] Device: {self.device}")
    
    def _create_model(self):
        """Create model"""
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
        else:
            return UNet(
                in_channels=self.config.get('data', {}).get('in_channels', 3),
                out_channels=self.config.get('data', {}).get('out_channels', 1),
                base_filters=self.config.get('model', {}).get('base_filters', 64)
            )
    
    def preprocess(self, image):
        """Preprocess image for inference"""
        transform = get_test_transforms(self.image_size)
        
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(str(image)).convert('RGB'))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        augmented = transform(image=image, mask=np.zeros((image.shape[0], image.shape[1])))
        return augmented['image'].unsqueeze(0)
    
    @torch.no_grad()
    def predict(self, image, threshold=0.5):
        """Predict segmentation mask"""
        # Handle path input
        if isinstance(image, (str, Path)):
            image = str(image)
            image = np.array(Image.open(image).convert('RGB'))
        
        image_tensor = self.preprocess(image).to(self.device)
        
        output = self.model(image_tensor)
        prob = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (prob > threshold).astype(np.uint8)
        
        return mask, prob
    
    @torch.no_grad()
    def predict_batch(self, image_paths, threshold=0.5):
        """Predict segmentation masks for a batch of images"""
        results = []
        
        for path in tqdm(image_paths, desc='Predicting'):
            mask, prob = self.predict(path, threshold)
            results.append({
                'path': str(path),
                'mask': mask,
                'probability': prob
            })
        
        return results
    
    def save_result(self, image_path, mask, output_path, overlay=True):
        """Save segmentation result"""
        # Save mask
        mask_image = Image.fromarray(mask * 255)
        mask_image.save(output_path)
        
        if overlay:
            # Create overlay
            image = Image.open(image_path).convert('RGB')
            image = image.resize(mask_image.size)
            
            # Create colored mask
            colored_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)
            colored_mask[mask == 1] = [255, 0, 0, 128]  # Red with alpha
            
            overlay_image = Image.fromarray(colored_mask, 'RGBA')
            result = Image.alpha_composite(image.convert('RGBA'), overlay_image)
            
            overlay_path = str(output_path).replace('.png', '_overlay.png').replace('.jpg', '_overlay.png')
            result.save(overlay_path)
            
            return overlay_path
        
        return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run inference on images')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input image or directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    args = parser.parse_args()
    
    # Initialize engine
    engine = InferenceEngine(args.checkpoint)
    
    # Process input
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # If output is a file path (has extension), use it directly
    # Otherwise, treat as directory
    if output_path.suffix in ['.png', '.jpg', '.jpeg']:
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        single_file = True
    else:
        output_dir = output_path
        output_dir.mkdir(parents=True, exist_ok=True)
        single_file = False
    
    if input_path.is_file():
        # Single image
        mask, prob = engine.predict(input_path, args.threshold)
        if single_file:
            save_path = output_path
        else:
            save_path = output_dir / f'{input_path.stem}_mask.png'
        engine.save_result(input_path, mask, save_path, overlay=True)
        print(f"Saved: {save_path}")
    else:
        # Directory
        image_files = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
        
        for img_path in tqdm(image_files, desc='Processing'):
            mask, prob = engine.predict(img_path, args.threshold)
            save_path = output_dir / f'{img_path.stem}_mask.png'
            engine.save_result(img_path, mask, save_path, overlay=True)
        
        print(f"Processed {len(image_files)} images")


if __name__ == '__main__':
    main()
