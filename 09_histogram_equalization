import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------
# Grayscale Equalization
# -------------------
gray = cv2.imread('img/09_sea_gray.png', cv2.IMREAD_GRAYSCALE)
gray_eq = cv2.equalizeHist(gray)

# -------------------
# Color Equalization (Y channel in YUV)
# -------------------
color = cv2.imread('img/09_sea_color.png')
color_yuv = cv2.cvtColor(color, cv2.COLOR_BGR2YUV)
color_yuv[:, :, 0] = cv2.equalizeHist(color_yuv[:, :, 0])
color_eq = cv2.cvtColor(color_yuv, cv2.COLOR_YUV2BGR)

# Convert color images to RGB for display
color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
color_eq_rgb = cv2.cvtColor(color_eq, cv2.COLOR_BGR2RGB)

# -------------------
# Display results
# -------------------
titles = ['Grayscale Original', 'Grayscale Equalized',
          'Color Original', 'Color Equalized (Y channel)']
images = [gray, gray_eq, color_rgb, color_eq_rgb]

plt.figure(figsize=(12, 8))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    cmap = 'gray' if i < 2 else None
    plt.imshow(images[i], cmap=cmap)
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
