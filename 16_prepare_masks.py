import cv2
import numpy as np
import glob
import os

# Get all PNG files starting with "16_"
mask_files = glob.glob(r"img\16_*.png")

for file in mask_files:
    print(f"Processing {file} ...")
    
    # Load the mask image
    img = cv2.imread(file)
    if img is None:
        print(f"Failed to load {file}")
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create a binary mask by thresholding (assuming bright background)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Create inverse mask
    mask_inv = cv2.bitwise_not(mask)

    # Apply the mask to isolate the face region
    masked_face = cv2.bitwise_and(img, img, mask=mask)

    # Save the outputs
    base = os.path.splitext(file)[0]
    cv2.imwrite(f"{base}_clean.png", masked_face)

print("✅ All masks processed and saved successfully.")
