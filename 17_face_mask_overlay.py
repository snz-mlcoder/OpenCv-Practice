import cv2
import numpy as np

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
if face_cascade.empty():
    raise IOError('Failed to load Haar cascade')

# Load your mask
face_mask = cv2.imread(r'img\16_maskera_clean.png')
if face_mask is None:
    raise IOError('Failed to load mask image')

# Webcam setup
cap = cv2.VideoCapture(0)
scaling_factor = 1.0

# Initial mask offset
offset_x = 0
offset_y = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in face_rects:
        h_new, w_new = int(1.4 * h), int(1.0 * w)

        # Apply offsets for fine-tuning
        x_new = x + offset_x
        y_new = int(y - 0.1 * h) + offset_y

        frame_h, frame_w = frame.shape[:2]
        if y_new < 0: y_new = 0
        if y_new + h_new > frame_h: h_new = frame_h - y_new
        if x_new < 0: x_new = 0
        if x_new + w_new > frame_w: w_new = frame_w - x_new

        face_mask_resized = cv2.resize(face_mask, (w_new, h_new), interpolation=cv2.INTER_AREA)
        frame_roi = frame[y_new:y_new + h_new, x_new:x_new + w_new]

        gray_mask = cv2.cvtColor(face_mask_resized, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_mask, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        masked_face = cv2.bitwise_and(face_mask_resized, face_mask_resized, mask=mask)
        masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
        combined = cv2.add(masked_face, masked_frame)
        frame[y_new:y_new + h_new, x_new:x_new + w_new] = combined

    cv2.imshow('Mask Overlay (Use arrow keys to adjust (WASD))', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break
    elif key == ord('w') or key == 82:  # up arrow or W
        offset_y -= 5
    elif key == ord('s') or key == 84:  # down arrow or S
        offset_y += 5
    elif key == ord('a') or key == 81:  # left arrow or A
        offset_x -= 5
    elif key == ord('d') or key == 83:  # right arrow or D
        offset_x += 5

cap.release()
cv2.destroyAllWindows()
