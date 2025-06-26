import cv2
import numpy as np

# Load and resize the image
img = cv2.imread('img/07_flower.png')
img = cv2.resize(img, (600, 400))
rows, cols = img.shape[:2]

# Callback for trackbars
def nothing(x):
    pass

# Create window and trackbars
cv2.namedWindow('Vignette Shifted Interactive')
cv2.createTrackbar('Sigma', 'Vignette Shifted Interactive', 100, 300, nothing)
cv2.createTrackbar('Shift X (%)', 'Vignette Shifted Interactive', 50, 100, nothing)
cv2.createTrackbar('Shift Y (%)', 'Vignette Shifted Interactive', 50, 100, nothing)

while True:
    sigma = max(1, cv2.getTrackbarPos('Sigma', 'Vignette Shifted Interactive'))

    shift_x_pct = cv2.getTrackbarPos('Shift X (%)', 'Vignette Shifted Interactive') / 100
    shift_y_pct = cv2.getTrackbarPos('Shift Y (%)', 'Vignette Shifted Interactive') / 100

    # Create an oversized Gaussian kernel
    kernel_x = cv2.getGaussianKernel(int(1.5 * cols), sigma)
    kernel_y = cv2.getGaussianKernel(int(1.5 * rows), sigma)
    kernel = kernel_y * kernel_x.T
    mask = kernel / np.max(kernel)

    # Crop the mask to shift the center
    start_x = int(shift_x_pct * 0.5 * cols)
    start_y = int(shift_y_pct * 0.5 * rows)
    mask = mask[start_y:start_y + rows, start_x:start_x + cols]

    # Apply mask to all 3 channels
    output = np.copy(img)
    for i in range(3):
        output[:, :, i] = output[:, :, i] * mask

    output = np.clip(output, 0, 255).astype(np.uint8)

    # Show result
    cv2.imshow('Vignette Shifted Interactive', output)

    key = cv2.waitKey(10) & 0xFF
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()
