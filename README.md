# OpenCV Practice

This repository contains small practice scripts and notebooks for learning and experimenting with OpenCV.

Each script focuses on a specific concept or technique such as filtering, edge detection, blurring, or color space manipulation.

## Contents

| # | Title | Description |
|---|-------|-------------|
| 01 | `01-basic-filters.py` | Applies identity, 3x3 average, and 5x5 average filters to an image using OpenCV and displays results with matplotlib. |
| 02 | `02_edge_sobel_laplacian.py` | Applies Sobel (x and y) and Laplacian filters to detect edges, shown in a 2x2 matplotlib grid. |
| 03 | `03_edge_canny.py`          | Uses the Canny edge detector with adjustable thresholds on a detailed image to highlight strong and weak edges. |
| 04 | `04_filter_blur_sharpen.py` | Applies motion blur and three sharpening filters using convolution kernels. Displays all results in a 2x3 matplotlib grid for visual comparison. |

| ... | ... | ... |

## How to Run

1. Make sure you have Python installed.
2. Install required packages:

   ```bash
   pip install opencv-python matplotlib numpy
