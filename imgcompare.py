#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
# "imagehash",
# "pillow",
# "numpy",
# "opencv-python"
# ]
# ///
"""
Dependencies:
uv pip install Pillow imagehash numpy opencv-python
"""

import os
import sys
from PIL import Image
import imagehash
import cv2
import numpy as np
from pathlib import Path
import subprocess
from typing import Dict, Tuple, List
import tempfile

def compute_image_hashes(image_path: str) -> Tuple[imagehash.ImageHash, imagehash.ImageHash, imagehash.ImageHash]:
    """Compute multiple perceptual hashes for better matching."""
    with Image.open(image_path) as img:
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Compute different types of hashes for better matching
        avg_hash = imagehash.average_hash(img)
        phash = imagehash.phash(img)
        dhash = imagehash.dhash(img)
        
        return avg_hash, phash, dhash

def calculate_similarity(img1_path: str, img2_path: str) -> float:
    """Calculate image similarity using perceptual hashing and structural similarity."""
    try:
        # Calculate perceptual hash similarity
        hash1 = compute_image_hashes(img1_path)
        hash2 = compute_image_hashes(img2_path)
        
        # Get minimum hash difference across different hash types
        hash_diffs = [h1 - h2 for h1, h2 in zip(hash1, hash2)]
        hash_similarity = 1 - (min(hash_diffs) / 64.0)  # Normalize to 0-1
        
        # Load images for structural comparison
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            return 1 - hash_similarity  # Return hash-only similarity
            
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Resize to similar dimensions for comparison
        height = min(gray1.shape[0], gray2.shape[0])
        if height > 500:  # Cap size for performance
            height = 500
        aspect1 = gray1.shape[1] / gray1.shape[0]
        aspect2 = gray2.shape[1] / gray2.shape[0]
        
        width1 = int(height * aspect1)
        width2 = int(height * aspect2)
        
        gray1 = cv2.resize(gray1, (width1, height))
        gray2 = cv2.resize(gray2, (width2, height))
        
        # If aspects are too different, likely not the same image
        if abs(aspect1 - aspect2) > 0.5:
            return 1.0
            
        # Pad smaller width to match
        if width1 != width2:
            max_width = max(width1, width2)
            if width1 < max_width:
                gray1 = cv2.copyMakeBorder(gray1, 0, 0, 0, max_width - width1, cv2.BORDER_CONSTANT, value=0)
            else:
                gray2 = cv2.copyMakeBorder(gray2, 0, 0, 0, max_width - width2, cv2.BORDER_CONSTANT, value=0)
        
        # Calculate structural similarity
        score = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)[0][0]
        structural_similarity = (score + 1) / 2  # Normalize to 0-1
        
        # Combine metrics (weighted average)
        final_similarity = (hash_similarity * 0.6 + structural_similarity * 0.4)
        return 1 - final_similarity  # Convert to distance metric (0 = identical)
        
    except Exception as e:
        print(f"Error in similarity calculation: {e}")
        return 1.0  # Return maximum difference on error

def find_matching_images(base_dir: str, target_dir: str, threshold: float = 0.3) -> List[Tuple[str, str, float]]:
    """Find matching images between directories."""
    matches = []
    
    target_files = list(Path(target_dir).rglob("*"))
    total_files = len(target_files)
    
    print(f"\nSearching through {total_files} files...")
    
    for i, target_file in enumerate(target_files, 1):
        if i % 10 == 0:
            print(f"Progress: {i}/{total_files} files processed", end='\r')
            
        if target_file.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.gif', '.webp'}:
            continue
            
        best_match = None
        best_similarity = float('inf')
        
        for base_file in Path(base_dir).rglob("*"):
            if base_file.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.gif', '.webp'}:
                continue
                
            try:
                similarity = calculate_similarity(str(target_file), str(base_file))
                if similarity < threshold and similarity < best_similarity:
                    best_similarity = similarity
                    best_match = base_file
            except Exception as e:
                print(f"Error comparing {target_file} and {base_file}: {e}")
        
        if best_match:
            matches.append((str(target_file), str(best_match), best_similarity))
    
    return sorted(matches, key=lambda x: x[2])

def create_side_by_side_image(img1_path: str, img2_path: str) -> str:
    """Create a temporary side-by-side image for viu display."""
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    
    # Calculate dimensions for side-by-side display
    max_height = max(img1.height, img2.height)
    scale1 = max_height / img1.height
    scale2 = max_height / img2.height
    
    new_width1 = int(img1.width * scale1)
    new_width2 = int(img2.width * scale2)
    
    img1 = img1.resize((new_width1, max_height))
    img2 = img2.resize((new_width2, max_height))
    
    # Create new image
    combined = Image.new('RGB', (new_width1 + new_width2, max_height))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (new_width1, 0))
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    combined.save(temp_file.name)
    return temp_file.name

def display_images(target_path: str, base_path: str):
    """Display images side by side using viu."""
    try:
        combined_image = create_side_by_side_image(target_path, base_path)
        subprocess.run(['viu', combined_image], check=True)
        os.unlink(combined_image)
    except subprocess.CalledProcessError:
        print("Error: viu not installed or failed to display images")
        sys.exit(1)

def replace_image(target_path: str, base_path: str):
    """Replace target image with base image."""
    import shutil
    backup_path = target_path + '.backup'
    shutil.copy2(target_path, backup_path)
    shutil.copy2(base_path, target_path)
    print(f"Replaced {target_path} with {base_path}")
    print(f"Backup saved as {backup_path}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <target_dir> <base_dir>")
        sys.exit(1)

    target_dir = sys.argv[1]
    base_dir = sys.argv[2]

    print("Finding matches...")
    matches = find_matching_images(base_dir, target_dir)
    
    if not matches:
        print("No matching images found")
        return

    for target_path, base_path, similarity in matches:
        print(f"\nPotential match (similarity score: {similarity:.3f}):")
        print(f"Target: {target_path}")
        print(f"Base: {base_path}")
        
        display_images(target_path, base_path)
        
        while True:
            choice = input("\nOptions: [s]kip, [r]eplace, [q]uit: ").lower()
            if choice == 's':
                break
            elif choice == 'r':
                replace_image(target_path, base_path)
                break
            elif choice == 'q':
                return
            else:
                print("Invalid choice")

if __name__ == "__main__":
    main()
