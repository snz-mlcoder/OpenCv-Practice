import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread(r'img/04_tree.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------------------
# Motion Blur Filter
# ---------------------
size = 15
kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size
motion_blur = cv2.filter2D(img, -1, kernel_motion_blur)
motion_blur_rgb = cv2.cvtColor(motion_blur, cv2.COLOR_BGR2RGB)

# ---------------------
# Sharpening Filters
# ---------------------
kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
kernel_sharpen_2 = np.array([[1,1,1], [1,-7,1], [1,1,1]])
kernel_sharpen_3 = np.array([
    [-1,-1,-1,-1,-1],
    [-1, 2, 2, 2,-1],
    [-1, 2, 8, 2,-1],
    [-1, 2, 2, 2,-1],
    [-1,-1,-1,-1,-1]
]) / 8.0

sharp_1 = cv2.filter2D(img, -1, kernel_sharpen_1)
sharp_2 = cv2.filter2D(img, -1, kernel_sharpen_2)
sharp_3 = cv2.filter2D(img, -1, kernel_sharpen_3)

# Convert sharpened outputs to RGB
sharp_1_rgb = cv2.cvtColor(sharp_1, cv2.COLOR_BGR2RGB)
sharp_2_rgb = cv2.cvtColor(sharp_2, cv2.COLOR_BGR2RGB)
sharp_3_rgb = cv2.cvtColor(sharp_3, cv2.COLOR_BGR2RGB)

# ---------------------
# Plot results
# ---------------------
titles = ['Original', 'Motion Blur', 'Sharpening 1', 'Sharpening 2', 'Edge Enhancement']
images = [img_rgb, motion_blur_rgb, sharp_1_rgb, sharp_2_rgb, sharp_3_rgb]

plt.figure(figsize=(15, 6))
for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
