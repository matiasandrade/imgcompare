#!/usr/bin/env -S uv run --script
# /// script
# requires-python = '>=3.10,<3.11'
# dependencies = [
# "imagededup",
# "numpy",
# "pillow",
# "typing"
# ]
# ///

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from imagededup.methods import CNN
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}


def get_similarity_dict(
    base_dir: Path, target_dir: Path
) -> Dict[np.str_, List[Tuple[np.str_, np.float32]]]:
    cnn_method = CNN()

    # Create temporary directory to hold all images
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Copy images from base and target dirs to temp dir
        # Add prefixes to track source directory
        for img in base_dir.glob("*"):
            if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS:
                shutil.copy2(img, temp_path / f"base_{img.name}")
        for img in target_dir.glob("*"):
            if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS:
                shutil.copy2(img, temp_path / f"target_{img.name}")

        # Find duplicates across all images
        duplicates = cnn_method.find_duplicates(
            image_dir=str(temp_path),
            scores=True,
            num_enc_workers=24,
        )

        # Filter results to only include base-target matches
        filtered_duplicates = {}
        for filename, matches in duplicates.items():
            if not isinstance(matches, list):
                matches = matches.items()

            # Only process if this is a base file
            if filename.startswith("base_"):
                base_matches = []
                for match, score in matches:
                    # Only include matches with target files
                    if match.startswith("target_"):
                        base_matches.append(
                            (match[7:], score)
                        )  # Remove "target_" prefix

                if base_matches:
                    filtered_duplicates[filename[5:]] = dict(
                        base_matches
                    )  # Remove "base_" prefix

        return filtered_duplicates


def create_side_by_side_image(img1_path: str, img2_path: str) -> str:
    with Image.open(img1_path) as img1, Image.open(img2_path) as img2:
        # Calculate target dimensions to make images roughly equal area
        img1_area = img1.width * img1.height
        img2_area = img2.width * img2.height
        
        # Scale the larger image down to match area of smaller image
        if img1_area > img2_area:
            scale = (img2_area / img1_area) ** 0.5
            new_width = int(img1.width * scale)
            new_height = int(img1.height * scale)
            img1 = img1.resize((new_width, new_height), Image.Resampling.LANCZOS)
        elif img2_area > img1_area:
            scale = (img1_area / img2_area) ** 0.5
            new_width = int(img2.width * scale)
            new_height = int(img2.height * scale)
            img2 = img2.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Calculate total width and height with extra space for text
        total_width = img1.width + img2.width
        max_height = max(img1.height, img2.height)
        text_height = 50  # Increased space for larger text
        combined = Image.new(
            "RGB", (total_width, max_height + text_height), color="white"
        )

        # Paste images
        combined.paste(img1, (0, 0))
        combined.paste(img2, (img1.width, 0))

        # Add text labels
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(combined)

        # Try to use a default font, fallback to default
        try:
            font = ImageFont.truetype(
               "/usr/share/fonts/TTF/DejaVuSansMono.ttf", 36  # Increased font size
            )
        except IOError:
            font = ImageFont.load_default()

        # Draw "Base" text under first image
        draw.text(
            (img1.width // 2 - 36, max_height + 5), "Base", fill="black", font=font
        )

        # Draw "Target" text under second image
        draw.text(
            (img1.width + img2.width // 2 - 36, max_height + 5),
            "Target",
            fill="black",
            font=font,
        )

        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        combined.save(temp_file.name)
        return temp_file.name


def move_to_low_res(target_path: str, base_path: Path, platform: str):
    target_path = Path(target_path)
    archive_dir = target_path.parent / "low_res_archive"
    archive_dir.mkdir(exist_ok=True)

    new_name = f"source_{platform}_{base_path.name}_{target_path.name}"
    archive_path = archive_dir / new_name
    shutil.move(target_path, archive_path)
    print(f"Moved to archive directory: {archive_path}")


def insert_image_alongside(target_path: str, base_path: Path, platform: str):
    """
    Insert the base image alongside the target image in the target directory.
    Renames the base image to avoid conflicts and preserve the original filename.
    """
    target_path = Path(target_path)
    insert_dir = target_path.parent

    # Create a unique filename for the inserted image
    new_name = f"inserted_{platform}_{base_path.name}"
    insert_path = insert_dir / new_name

    # Copy the base image to the target directory with the new name
    shutil.copy2(base_path, insert_path)
    print(f"Inserted image alongside {target_path}: {insert_path}")


def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <target_dir> <base_dir> <platform>")
        sys.exit(1)

    target_dir = Path(sys.argv[1])
    base_dir = Path(sys.argv[2])
    platform = sys.argv[3]

    print("Finding matches...")
    similarity_dict = get_similarity_dict(base_dir, target_dir)

    if not similarity_dict:
        print("No matching images found")
        return
    for base_img, matches in similarity_dict.items():
        for target_img, similarity in matches.items():
            print(f"\nPotential match (similarity score: {similarity:.3f}):")
            base_path = base_dir / base_img
            target_path = target_dir / target_img
            print(f"Target: {target_path}")
            print(f"Base: {base_path}")

            try:
                combined_image = create_side_by_side_image(
                    str(target_path), str(base_path)
                )
                subprocess.run(["viu", combined_image], check=True)
                os.unlink(combined_image)
            except subprocess.CalledProcessError:
                print("Error: viu not installed or failed to display images")

            while True:
                choice = input(
                    "\nOptions: [s]kip, [r]eplace, [i]nsert, [q]uit: "
                ).lower()
                if choice == "s":
                    break
                elif choice == "r":
                    move_to_low_res(str(target_path), base_path, platform)
                    shutil.copy2(base_path, target_path)
                    print(f"Replaced {target_path} with {base_path}")
                    break
                elif choice == "i":
                    insert_image_alongside(str(target_path), base_path, platform)
                    break
                elif choice == "q":
                    return
                else:
                    print("Invalid choice")


if __name__ == "__main__":
    main()
