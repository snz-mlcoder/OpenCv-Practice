import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Check if cascades loaded properly
if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')
if eye_cascade.empty():
    raise IOError('Unable to load the eye cascade classifier xml file')

# Load main image and sunglasses image
img = cv2.imread(r'img\19_face.jpg')
sunglasses_img = cv2.imread(r'img\19_sunglasses.png')

if img is None or sunglasses_img is None:
    raise IOError('Failed to load one or both images.')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)

    centers = []
    for (x_eye, y_eye, w_eye, h_eye) in eyes:
        centers.append((x + int(x_eye + 0.5 * w_eye), y + int(y_eye + 0.5 * h_eye)))

    if len(centers) >= 2:
        # Sort eyes left to right
        centers = sorted(centers, key=lambda pt: pt[0])
        sunglasses_width = int(2.12 * abs(centers[1][0] - centers[0][0]))

        # Resize sunglasses
        h_sun, w_sun = sunglasses_img.shape[:2]
        scaling_factor = sunglasses_width / w_sun
        overlay_sunglasses = cv2.resize(sunglasses_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        # Positioning
        x_min = centers[0][0] - int(0.26 * overlay_sunglasses.shape[1])
        y_min = centers[0][1] - int(0.5 * overlay_sunglasses.shape[0])

        # Ensure within image boundaries
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_min + overlay_sunglasses.shape[1], img.shape[1])
        y_max = min(y_min + overlay_sunglasses.shape[0], img.shape[0])
        overlay_sunglasses = overlay_sunglasses[:y_max - y_min, :x_max - x_min]

        # Prepare mask
        overlay_img = np.ones_like(img) * 255
        overlay_img[y_min:y_max, x_min:x_max] = overlay_sunglasses

        gray_overlay = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_overlay, 110, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Apply masks
        temp = cv2.bitwise_and(img, img, mask=mask)
        temp2 = cv2.bitwise_and(overlay_img, overlay_img, mask=mask_inv)
        final_img = cv2.add(temp, temp2)

        break  # Only process first detected face

# Display results
# Display results
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(final_img_rgb)
plt.title('With Sunglasses')
plt.axis('off')

plt.tight_layout()
plt.show()


