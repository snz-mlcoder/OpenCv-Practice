import cv2
import numpy as np

# Load the image
img = cv2.imread('img/07_flower.png')
img = cv2.resize(img, (600, 400))  
rows, cols = img.shape[:2]

# Callback for trackbar 
def nothing(x):
    pass

# Create window and trackbar
cv2.namedWindow('Vignette Interactive')
cv2.createTrackbar('Sigma', 'Vignette Interactive', 10, 300, nothing)

while True:
    sigma = cv2.getTrackbarPos('Sigma', 'Vignette Interactive')
    sigma = max(1, sigma)  # prevent division by zero or too small

    # Generate Gaussian kernels
    kernel_x = cv2.getGaussianKernel(cols, sigma)
    kernel_y = cv2.getGaussianKernel(rows, sigma)
    kernel = kernel_y * kernel_x.T

    # Normalize the mask to [0, 1]
    mask = kernel / np.max(kernel)

    # Apply the mask to all 3 channels
    output = np.copy(img)
    for i in range(3):
        output[:, :, i] = output[:, :, i] * mask

    output = np.clip(output, 0, 255).astype(np.uint8)

    # Show result
    cv2.imshow('Vignette Interactive', output)

    # Exit on ESC
    key = cv2.waitKey(10) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()
