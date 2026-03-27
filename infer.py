"""
Inference Script for Medical Image Segmentation
With edge smoothing and original image size preservation.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm

from models import UNet, AttentionUNet, AttentionUNetLite, EnhancedAttentionUNet


class InferenceEngine:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = self.checkpoint.get("config", {})
        
        self.model = self._create_model()
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.model_image_size = tuple(self.config.get("data", {}).get("image_size", [224, 224]))
        
        print(f"[INFO] Model loaded: {checkpoint_path}")
        print(f"[INFO] Device: {self.device}")
    
    def _create_model(self):
        model_name = self.config.get("model", {}).get("name", "unet")
        
        if model_name == "attention_unet":
            return AttentionUNet(3, 1, 64)
        elif model_name == "attention_unet_lite":
            return AttentionUNetLite(3, 1, 32)
        elif model_name == "enhanced_attention_unet":
            return EnhancedAttentionUNet(3, 1, 64, deep_supervision=False)
        else:
            return UNet(3, 1, 64)
    
    def preprocess(self, image, target_size=None):
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(str(image)).convert("RGB"))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        
        h, w = image.shape[:2]
        target_h, target_w = target_size or self.model_image_size
        
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                     cv2.BORDER_CONSTANT, value=[124, 117, 104])
        
        padded = padded.astype(np.float32) / 255.0
        padded = (padded - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        padded = padded.transpose(2, 0, 1)
        padded = torch.from_numpy(padded).float().unsqueeze(0)
        
        return padded, (h, w), (new_h, new_w), (top, left)
    
    def postprocess(self, mask, original_size, padded_size, pad_info):
        original_h, original_w = original_size
        padded_h, padded_w = padded_size
        top, left = pad_info
        
        mask_cropped = mask[top:top+padded_h, left:left+padded_w]
        mask_resized = cv2.resize(mask_cropped, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        
        return mask_resized
    
    def smooth_edges(self, mask, kernel_size=5):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_smooth = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_OPEN, kernel)
        return mask_smooth
    
    @torch.no_grad()
    def predict(self, image, threshold=0.5, smooth_edges=True, kernel_size=5, keep_original_size=True):
        if isinstance(image, (str, Path)):
            orig_image = Image.open(str(image)).convert("RGB")
            orig_size = orig_image.size[::-1]
        else:
            orig_image = image.convert("RGB")
            orig_size = orig_image.size[::-1]
        
        np_image = np.array(orig_image)
        
        tensor, (orig_h, orig_w), (new_h, new_w), pad_info = self.preprocess(np_image)
        tensor = tensor.to(self.device)
        
        output = self.model(tensor)
        prob = torch.sigmoid(output).squeeze().cpu().numpy()
        
        if prob.ndim == 4:
            prob = prob[0]
        
        mask = (prob > threshold).astype(np.uint8)
        
        if keep_original_size:
            mask = self.postprocess(mask, (orig_h, orig_w), (new_h, new_w), pad_info)
        
        if smooth_edges:
            mask = self.smooth_edges(mask, kernel_size=kernel_size)
        
        return mask, prob
    
    def save_result(self, image_path, mask, output_path, overlay=True):
        mask_image = Image.fromarray(mask * 255)
        mask_image.save(output_path)
        
        if overlay:
            image = Image.open(image_path).convert("RGB")
            image = image.resize(mask_image.size)
            
            colored_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)
            colored_mask[mask == 1] = [255, 0, 0, 128]
            
            overlay_image = Image.fromarray(colored_mask, "RGBA")
            result = Image.alpha_composite(image.convert("RGBA"), overlay_image)
            
            overlay_path = str(output_path).replace(".png", "_overlay.png").replace(".jpg", "_overlay.png")
            result.save(overlay_path)
            
            return overlay_path
        
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Run inference on images")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-smooth", action="store_true")
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--no-resize", action="store_true")
    args = parser.parse_args()
    
    engine = InferenceEngine(args.checkpoint)
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    single_file = output_path.suffix in [".png", ".jpg", ".jpeg"]
    output_dir = output_path.parent if single_file else output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    smooth_edges = not args.no_smooth
    keep_original_size = not args.no_resize
    
    print(f"[CONFIG] threshold={args.threshold}, smooth={smooth_edges}, kernel={args.kernel_size}")
    
    if input_path.is_file():
        mask, prob = engine.predict(input_path, args.threshold, smooth_edges, args.kernel_size, keep_original_size)
        
        if single_file:
            save_path = output_path
        else:
            save_path = output_dir / f"{input_path.stem}_mask.png"
        
        engine.save_result(input_path, mask, save_path, overlay=True)
        print(f"Saved: {save_path}")
    else:
        for img_path in tqdm(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))):
            mask, prob = engine.predict(img_path, args.threshold, smooth_edges, args.kernel_size, keep_original_size)
            save_path = output_dir / f"{img_path.stem}_mask.png"
            engine.save_result(img_path, mask, save_path, overlay=True)


if __name__ == "__main__":
    main()
