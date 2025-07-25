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
| 05 | `05_filter_emboss.py` | Applies embossing filters in different directions (South West, South East, North West) and shifts the intensity for visualization. |
| 06 | `06_morphology_erosion_dilation.py` | Demonstrates erosion and dilation on binary text image using a 5x5 kernel. Useful for shrinking or expanding foreground pixels. |
| 07 | `07_filter_vignette_interactive.py` | Interactive vignette filter using a trackbar to adjust Gaussian sigma in real-time. |
| 08 | `08_filter_vignette_shifted.py` | Applies a vignette effect with a shifted center, simulating off-center focus using cropped Gaussian mask. |
| 09 | `09_histogram_equalization_color.py` | Improves contrast of a color image by applying histogram equalization on the Y channel in YUV color space. |
| 10 | `10_webcam_color_space_switch.py` | Captures webcam feed and switches color spaces (Grayscale, YUV, HSV) interactively using keyboard keys. |
| 11 | `11_mouse_quadrant_highlight.py` | Detects mouse click on an image and highlights the clicked quadrant (top-left, top-right, bottom-left, bottom-right). |
| 12 | `12_webcam_negative_roi.py` | Live webcam stream where user can select a region with the mouse, and apply a "negative film" effect to the selected ROI. |
| 13 | `13_webcam_cartoonize.py` | Converts live webcam stream to cartoon-style or sketch-style using Laplacian edge detection and bilateral filtering. Use keys: 's' for sketch, 'c' for color, ESC to exit. |
| 14 | `14_filter_comparison.py` | Compare Gaussian Blur and Bilateral Filter on an image. Shows side-by-side differences in edge preservation and smoothing. |
| 15 | `15_face_detection.py` | Real-time face detection using Haar Cascade and webcam. Automatically counts and saves detected faces. Press ESC to exit. |
| 16 | `16_prepare_masks.py` | Batch process all mask images starting with `16_` to remove background and save clean versions for face overlay. |
| 17 | `17_face_mask_overlay.py` | Real-time face mask overlay using Haar Cascade. Switch masks with keys (1/2/3) and adjust position using arrow keys (WASD). |
| 18 | `18_eye_detector.py | Real-time eye detection using Haar cascades. Draws green circles around detected eyes in the webcam feed. |
| 19 | `19_face_sunglasses_overlay.py` | Adds sunglasses to a face image using Haar cascades (face + eye). Includes matplotlib comparison plot. |
| 20 | `20_face_parts_detection.py` | Real-time detection of face parts (eyes, nose, mouth) using Haar Cascades and webcam. Results are shown live with rectangles on each detected part. |


| ... | ... | ... |

## How to Run

1. Make sure you have Python installed.
2. Install required packages:

   ```bash
   pip install opencv-python matplotlib numpy
