# 03_edge_canny.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image in grayscale
img = cv2.imread(r'img\03_train.png', cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detector
canny = cv2.Canny(img, 50, 240)

# Plot original and Canny result
titles = ['Original', 'Canny Edge Detection']
images = [img, canny]

plt.figure(figsize=(10, 4))
for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
