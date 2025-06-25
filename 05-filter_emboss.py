import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the input image
img = cv2.imread(r'img/05_house.png')

# Convert to grayscale for embossing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define emboss kernels
kernel_emboss_1 = np.array([[0, -1, -1],
                            [1,  0, -1],
                            [1,  1,  0]])

kernel_emboss_2 = np.array([[-1, -1, 0],
                            [-1,  0, 1],
                            [ 0,  1, 1]])

kernel_emboss_3 = np.array([[ 1,  0,  0],
                            [ 0,  0,  0],
                            [ 0,  0, -1]])

# Apply emboss filters and add 128 to shift intensity
output_1 = cv2.filter2D(gray, -1, kernel_emboss_1) + 128
output_2 = cv2.filter2D(gray, -1, kernel_emboss_2) + 128
output_3 = cv2.filter2D(gray, -1, kernel_emboss_3) + 128

# Plot original and embossed images
titles = ['Original', 'Emboss - South West', 'Emboss - South East', 'Emboss - North West']
images = [gray, output_1, output_2, output_3]

plt.figure(figsize=(12, 6))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
