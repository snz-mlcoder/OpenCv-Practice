import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('img/14_carpet.png') 
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply Gaussian Blur
img_gaussian = cv2.GaussianBlur(img, (13, 13), 0)
img_gaussian_rgb = cv2.cvtColor(img_gaussian, cv2.COLOR_BGR2RGB)

# Apply Bilateral Filter
img_bilateral = cv2.bilateralFilter(img, d=20, sigmaColor=70, sigmaSpace=50)
img_bilateral_rgb = cv2.cvtColor(img_bilateral, cv2.COLOR_BGR2RGB)

# Plot results
titles = ['Original', 'Gaussian Blur', 'Bilateral Filter']
images = [img_rgb, img_gaussian_rgb, img_bilateral_rgb]

plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
