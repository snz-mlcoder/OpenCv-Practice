import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image in grayscale mode
img = cv2.imread(r'img\02_input_shape.png', cv2.IMREAD_GRAYSCALE)

# Apply Sobel filters (horizontal and vertical edges)
sobel_horizontal = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobel_vertical = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

# Apply Laplacian filter (second derivative in both directions)
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# Plot all results in a 2x2 grid
titles = ['Original', 'Sobel Horizontal', 'Sobel Vertical', 'Laplacian']
images = [img, sobel_horizontal, sobel_vertical, laplacian]

plt.figure(figsize=(10, 8))
for i in range(4):
    plt.subplot(2, 2, i + 1)  # 2 rows, 2 columns
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
