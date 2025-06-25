import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image from the local directory
img = cv2.imread(r'img\01.png')

# Convert BGR image to RGB for correct display with matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Define filters (kernels)
kernel_identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
kernel_3x3 = np.ones((3, 3), np.float32) / 9.0
kernel_5x5 = np.ones((5, 5), np.float32) / 25.0

# Apply filters using cv2.filter2D
output_identity = cv2.filter2D(img, -1, kernel_identity)
output_3x3 = cv2.filter2D(img, -1, kernel_3x3)
output_5x5 = cv2.filter2D(img, -1, kernel_5x5)

# Convert filtered images to RGB for matplotlib
output_identity_rgb = cv2.cvtColor(output_identity, cv2.COLOR_BGR2RGB)
output_3x3_rgb = cv2.cvtColor(output_3x3, cv2.COLOR_BGR2RGB)
output_5x5_rgb = cv2.cvtColor(output_5x5, cv2.COLOR_BGR2RGB)

# Display the original and filtered images using matplotlib
titles = ['Original', 'Identity Filter', '3x3 Filter', '5x5 Filter']
images = [img_rgb, output_identity_rgb, output_3x3_rgb, output_5x5_rgb]

plt.figure(figsize=(12, 6))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
