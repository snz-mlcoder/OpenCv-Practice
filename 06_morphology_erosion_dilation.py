import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the input image in grayscale
img = cv2.imread('img/06_text.png', cv2.IMREAD_GRAYSCALE)

# Threshold the image to binary (black & white)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Define a 5x5 kernel
kernel = np.ones((5, 5), np.uint8)

# Apply erosion and dilation
img_erosion = cv2.erode(binary, kernel, iterations=1)
img_dilation = cv2.dilate(binary, kernel, iterations=1)

# Display results using matplotlib
titles = ['Original', 'Binary', 'Erosion', 'Dilation']
images = [img, binary, img_erosion, img_dilation]

plt.figure(figsize=(12, 6))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
