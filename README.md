# Image Deduplication and Management Tool

A Python script for finding and managing similar images across directories using CNN-based image comparison. The tool helps identify potential duplicate or similar images and provides options to replace lower quality versions or manage image organization.

## Features

- CNN-based image similarity detection
- Side-by-side image comparison
- Interactive decision making for each match
- Support for multiple image formats (JPG, PNG, GIF, BMP, TIFF, WebP)
- Automatic image scaling for fair comparisons
- Archive system for replaced images

## Prerequisites

- Python 3.10
- `uv` package manager
- Terminal image viewer (`viu`)
- Required Python packages (automatically installed by uv):
  - imagededup
  - numpy
  - pillow
  - typing

## Installation

1. Ensure you have Python 3.10 and `uv` installed
2. Install the terminal image viewer:
   ```bash
   # For Ubuntu/Debian
   apt-get install viu
   # For macOS
   brew install viu
   ```
3. The script will automatically install required Python dependencies using `uv`

## Usage

```bash
python script.py <target_dir> <base_dir> <platform>
```

### Parameters

- `target_dir`: Directory containing images to be checked
- `base_dir`: Directory containing reference/source images
- `platform`: Platform identifier for archiving purposes

### Interactive Options

For each potential match, you'll be presented with these options:

- `s`: Skip this match
- `r`: Replace target image with base image (archives original)
- `i`: Insert base image alongside target image
- `q`: Quit the program

### Examples

```bash
python script.py ./downloads ./originals instagram
python script.py ./web_images ./high_res_sources facebook
```

## How It Works

1. The script creates temporary copies of images from both directories
2. Uses CNN-based comparison to find similar images
3. Shows side-by-side comparisons with similarity scores
4. Allows interactive decision making for each match
5. Manages image replacements and archiving

## Archive System

When replacing images, originals are moved to a `low_res_archive` directory with the following naming convention:
```
source_<platform>_<base_filename>_<target_filename>
```

## File Organization

When inserting images alongside existing ones, the new files are named:
```
inserted_<platform>_<base_filename>
```

## Limitations

- Requires the `viu` terminal image viewer for visual comparisons
- CPU-intensive for large image sets
- Memory usage scales with image size and quantity

## Error Handling

- Gracefully handles missing `viu` installation
- Validates command-line arguments
- Provides clear error messages for common issues

## Notes

- Images are automatically scaled for fair comparison
- Supports multiple image formats
- Uses system fonts for labels (falls back to default if unavailable)
- Temporary files are automatically cleaned up
