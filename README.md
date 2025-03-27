# POCvsOpenCV_StitchingCompare

A GUI application to compare two image stitching methods:
1. Phase Only Correlation (POC) - Frequency domain approach
2. OpenCV feature-based stitching (SIFT + homography)

## Overview

This application provides a visual comparison between frequency-domain and spatial-domain approaches to image stitching. It allows users to apply different levels of degradation (blur and noise) to test the robustness of each method.

![Application Screenshot Placeholder]()

## Features

- **Side-by-side comparison**: View the results of both stitching methods simultaneously
- **Degradation testing**: Apply Gaussian blur and noise to test algorithm robustness
- **Real-time processing**: See the effects of parameter changes immediately
- **Dark-themed UI**: Modern, eye-friendly interface for image processing

## Image Stitching Methods

### 1. Phase Only Correlation (POC)

Uses frequency domain analysis to find the translation between two images. This method:
- Works by calculating cross-power spectrum in Fourier domain
- Typically handles translation/shift but not rotation or scaling
- Often more robust to noise and illumination changes
- Implemented through the `openfv` package's `ww_phase_only_correlation` function

### 2. OpenCV Feature-based Stitching

Uses SIFT feature detection and matching to create a perspective transformation:
- Extracts keypoints and descriptors using SIFT
- Matches features using brute-force matching and Lowe's ratio test
- Computes homography matrix using RANSAC
- Performs perspective warping to align images
- Can handle more complex transformations (rotation, scaling, perspective)

## Requirements

- Python 3.6+
- NumPy
- OpenCV
- Matplotlib
- PIL (Pillow)
- Tkinter
- openfv (for Phase Only Correlation)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/bemoregt/POCvsOpenCV_StitchingCompare.git
cd POCvsOpenCV_StitchingCompare
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Install the openfv package (if not available via pip, check repository-specific instructions)

## Usage

1. Prepare two overlapping images for stitching
2. Update the image paths in the main section of `image_stitching_app.py`:
```python
img1_path = "path/to/your/first/image.png"
img2_path = "path/to/your/second/image.png"
```
3. Run the application:
```bash
python image_stitching_app.py
```
4. Adjust blur and noise parameters using the dropdown menus
5. Click "Stitch" to process the images and view the results

## Algorithm Comparison

| Aspect | POC Method | OpenCV Method |
|--------|------------|---------------|
| Domain | Frequency | Spatial |
| Transformation | Translation only | Full homography |
| Feature extraction | No | Yes (SIFT) |
| Robustness to noise | Higher | Moderate |
| Computational complexity | Lower | Higher |
| Works with rotation | No | Yes |

## Applications

This comparison is particularly useful for:
- Computer vision research
- Image processing education
- Evaluating algorithm performance in different conditions
- PCB inspection and industrial imaging
- Medical image registration

## License

[MIT License](LICENSE)

## Citation

If you use this code in your research, please cite:

```
@software{POCvsOpenCV_StitchingCompare,
  author = {Park, Gromit},
  title = {POCvsOpenCV_StitchingCompare: Image Stitching Comparison},
  url = {https://github.com/bemoregt/POCvsOpenCV_StitchingCompare},
  year = {2025}
}
```