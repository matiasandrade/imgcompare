#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
# "imagehash",
# "pillow",
# "numpy",
# "opencv-python"
# ]
# ///

import os
import sys
from PIL import Image
import imagehash
import cv2
import numpy as np
from pathlib import Path
import subprocess
from typing import Dict, Tuple, List, Optional
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class ImageMatch:
    target_path: str
    base_path: str
    similarity: float

class ImageHashCache:
    def __init__(self):
        self.cache: Dict[str, Tuple[imagehash.ImageHash, imagehash.ImageHash, imagehash.ImageHash]] = {}

    @lru_cache(maxsize=1000)
    def get_hashes(self, image_path: str) -> Tuple[imagehash.ImageHash, imagehash.ImageHash, imagehash.ImageHash]:
        if image_path not in self.cache:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                self.cache[image_path] = (
                    imagehash.average_hash(img),
                    imagehash.phash(img),
                    imagehash.dhash(img)
                )
        return self.cache[image_path]

def calculate_similarity(img1_path: str, img2_path: str, hash_cache: ImageHashCache) -> float:
    try:
        # Fast hash comparison first
        hash1 = hash_cache.get_hashes(img1_path)
        hash2 = hash_cache.get_hashes(img2_path)

        hash_diffs = [h1 - h2 for h1, h2 in zip(hash1, hash2)]
        hash_similarity = 1 - (min(hash_diffs) / 64.0)

        # Early exit if hash similarity is too low
        if hash_similarity < 0.5:
            return 1.0

        # Structural similarity for promising matches
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            return 1.0

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Check aspect ratios
        aspect1 = gray1.shape[1] / gray1.shape[0]
        aspect2 = gray2.shape[1] / gray2.shape[0]
        if abs(aspect1 - aspect2) > 0.5:
            return 1.0

        # Handle template size requirements
        if gray1.shape[0] > gray2.shape[0] or gray1.shape[1] > gray2.shape[1]:
            gray1, gray2 = gray2, gray1
        score = cv2.matchTemplate(gray2, gray1, cv2.TM_CCOEFF_NORMED)[0][0]
        structural_similarity = (score + 1) / 2

        return 1 - (hash_similarity * 0.4 + structural_similarity * 0.6)

    except Exception as e:
        print(f"Error comparing {img1_path} and {img2_path}: {e}")
        return 1.0

def process_target_file(args: Tuple[Path, List[Path], ImageHashCache, float]) -> Optional[ImageMatch]:
    target_file, base_files, hash_cache, threshold = args

    if target_file.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.gif', '.webp'}:
        return None

    best_match = None
    best_similarity = float('inf')

    for base_file in base_files:
        if base_file.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.gif', '.webp'}:
            continue

        similarity = calculate_similarity(str(target_file), str(base_file), hash_cache)
        if similarity < threshold and similarity < best_similarity:
            best_similarity = similarity
            best_match = base_file

    if best_match:
        return ImageMatch(str(target_file), str(best_match), best_similarity)
    return None

def find_matching_images(base_dir: str, target_dir: str, threshold: float = 0.3) -> List[ImageMatch]:
    target_files = list(Path(target_dir).rglob("*"))
    base_files = list(Path(base_dir).rglob("*"))
    hash_cache = ImageHashCache()

    with ThreadPoolExecutor() as executor:
        args = [(f, base_files, hash_cache, threshold) for f in target_files]
        matches = list(executor.map(process_target_file, args))

    return sorted([m for m in matches if m is not None], key=lambda x: x.similarity)

def create_side_by_side_image(img1_path: str, img2_path: str) -> str:
    with Image.open(img1_path) as img1, Image.open(img2_path) as img2:
        combined = Image.new('RGB', (img1.width + img2.width, max(img1.height, img2.height)))
        combined.paste(img1, (0, 0))
        combined.paste(img2, (img1.width, 0))

        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        combined.save(temp_file.name)
        return temp_file.name

def move_to_low_res(target_path: str):
    target_path = Path(target_path)
    low_res_dir = target_path.parent.parent / f"{target_path.parent.name}_low_res"
    low_res_dir.mkdir(exist_ok=True)

    backup_path = low_res_dir / target_path.name
    shutil.move(target_path, backup_path)
    print(f"Moved to low-res directory: {backup_path}")

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

    for match in matches:
        print(f"\nPotential match (similarity score: {match.similarity:.3f}):")
        print(f"Target: {match.target_path}")
        print(f"Base: {match.base_path}")

        try:
            combined_image = create_side_by_side_image(match.target_path, match.base_path)
            subprocess.run(['viu', combined_image], check=True)
            os.unlink(combined_image)
        except subprocess.CalledProcessError:
            print("Error: viu not installed or failed to display images")

        while True:
            choice = input("\nOptions: [s]kip, [r]eplace, [q]uit: ").lower()
            if choice == 's':
                break
            elif choice == 'r':
                move_to_low_res(match.target_path)
                shutil.copy2(match.base_path, match.target_path)
                print(f"Replaced {match.target_path} with {match.base_path}")
                break
            elif choice == 'q':
                return
            else:
                print("Invalid choice")

if __name__ == "__main__":
    main()
