import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image as imwrite
import os
import time
import re
from torchvision import transforms
from test_dataset import dehaze_test_dataset
from model import final_net
import torch.nn.functional as F
from PIL import Image
import numpy as np

def process_patch(model, patch, device):
    with torch.no_grad():
        patch = patch.to(device)
        return model(patch)

def split_image(img_tensor, window_size=512, overlap=64):
    """Split image into overlapping patches."""
    _, _, h, w = img_tensor.shape
    patches = []
    positions = []
    
    for y in range(0, h, window_size - overlap):
        for x in range(0, w, window_size - overlap):
            # Calculate patch boundaries
            y1 = y
            x1 = x
            y2 = min(y + window_size, h)
            x2 = min(x + window_size, w)
            
            # Extract patch
            patch = img_tensor[:, :, y1:y2, x1:x2]
            
            # Pad if necessary to maintain consistent size
            if patch.shape[2] < window_size or patch.shape[3] < window_size:
                ph = window_size - patch.shape[2]
                pw = window_size - patch.shape[3]
                patch = F.pad(patch, (0, pw, 0, ph))
            
            patches.append(patch)
            positions.append((y1, y2, x1, x2))
    
    return patches, positions

def merge_patches(patches, positions, original_size, window_size=512, overlap=64):
    """Merge overlapping patches with linear blending."""
    h, w = original_size
    result = torch.zeros((1, 3, h, w), device=patches[0].device)
    weights = torch.zeros((1, 3, h, w), device=patches[0].device)
    
    for patch, (y1, y2, x1, x2) in zip(patches, positions):
        # Calculate actual patch size (might be smaller at edges)
        patch_h = y2 - y1
        patch_w = x2 - x1
        
        # Extract the valid portion of the patch (without padding)
        valid_patch = patch[:, :, :patch_h, :patch_w]
        
        # Create weight mask
        weight = torch.ones_like(valid_patch)
        
        # Apply linear blending in overlap regions
        if overlap > 0:
            for i in range(overlap):
                weight_value = i / overlap
                # Blend left edge if not at image boundary
                if x1 > 0 and i < patch_w:
                    weight[:, :, :, i] *= weight_value
                # Blend right edge if not at image boundary
                if x2 < w and patch_w - i - 1 >= 0:
                    weight[:, :, :, -(i + 1)] *= weight_value
                # Blend top edge if not at image boundary
                if y1 > 0 and i < patch_h:
                    weight[:, :, i, :] *= weight_value
                # Blend bottom edge if not at image boundary
                if y2 < h and patch_h - i - 1 >= 0:
                    weight[:, :, -(i + 1), :] *= weight_value
        
        result[:, :, y1:y2, x1:x2] += valid_patch * weight
        weights[:, :, y1:y2, x1:x2] += weight
    
    # Normalize by weights to complete blending
    valid_mask = weights > 0
    result[valid_mask] = result[valid_mask] / weights[valid_mask]
    
    return result



def main():
    parser = argparse.ArgumentParser(description='Shadow')
    parser.add_argument('--input_dir', type=str, default='/ShadowDataset/test/')
    parser.add_argument('--output_dir', type=str, default='output/')
    parser.add_argument('--window_size', type=int, default=512, help='Size of sliding window')
    parser.add_argument('--overlap', type=int, default=64, help='Overlap between windows')
    args = parser.parse_args()

     # Ensure paths end with slash
    args.input_dir = os.path.join(args.input_dir, '')
    args.output_dir = os.path.join(args.output_dir, '')
    print('')
    print(f'input_dir: {args.input_dir}')
    print(f'output_dir: {args.output_dir}')
    print('')
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    test_dataset = dehaze_test_dataset(args.input_dir)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = final_net()
    
    try:
        model.remove_model.load_state_dict(torch.load(os.path.join('weights', 'shadowremoval.pkl'), map_location='cpu'), strict=True)
        print('Loading removal_model success')
    except:
        print('Loading removal_model error')
        return

    try:
        model.enhancement_model.load_state_dict(torch.load(os.path.join('weights', 'refinement.pkl'), map_location='cpu'), strict=True)
        print('Loading enhancement model success')
    except:
        print('Loading enhancement model error')
        return

    model = model.to(device)
    model.eval()
    
    total_time = 0
    with torch.no_grad():
        for batch_idx, (input_img, name) in enumerate(test_loader):
            print(f"Processing {name[0]}")
            
            # Get image dimensions
            _, _, h, w = input_img.shape
            print(f"Image size: {w}x{h}")
            
            # Split image into patches
            patches, positions = split_image(input_img, args.window_size, args.overlap)
            print(f"Split into {len(patches)} patches")
            
            # Process each patch
            processed_patches = []
            for i, patch in enumerate(patches):
                print(f"Processing patch {i+1}/{len(patches)}")
                processed_patch = process_patch(model, patch, device)
                processed_patches.append(processed_patch)
                torch.cuda.empty_cache()  # Clear GPU memory after each patch
            
            # Merge patches
            result = merge_patches(processed_patches, positions, (h, w), args.window_size, args.overlap)
            
            # Save result
            name = re.findall(r"\d+", str(name))
            save_path = os.path.join(args.output_dir, f"{name[0]}.png")
            print(f"Saving result to {save_path}")
            imwrite(result, save_path, range=(0, 1))
            
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
