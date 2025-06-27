import cv2
import numpy as np

# Cartoonization function
def cartoonize_image(img, ds_factor=4, sketch_mode=False):
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray, 7)

    # Edge detection using Laplacian and thresholding
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=5)
    ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)

    # Return sketch version
    if sketch_mode:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Downsample the image for faster bilateral filtering
    img_small = cv2.resize(img, None, fx=1.0/ds_factor, fy=1.0/ds_factor, interpolation=cv2.INTER_AREA)

    # Apply bilateral filter multiple times
    for i in range(10):
        img_small = cv2.bilateralFilter(img_small, d=5, sigmaColor=5, sigmaSpace=7)

    # Upsample back to original size
    img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_LINEAR)

    # Combine edge mask with filtered image
    return cv2.bitwise_and(img_output, img_output, mask=mask)

# Main program
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    cur_char = -1
    prev_char = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)

        # Handle keyboard input
        c = cv2.waitKey(1)
        if c == 27:  # ESC
            break
        if c > -1 and c != prev_char:
            cur_char = c
            prev_char = c

        # Choose mode based on key
        if cur_char == ord('s'):
            output = cartoonize_image(frame, sketch_mode=True)
        elif cur_char == ord('c'):
            output = cartoonize_image(frame, sketch_mode=False)
        else:
            output = frame

        # Show result
        cv2.imshow('Cartoonize [s: sketch | c: color | ESC: exit]', output)

    cap.release()
    cv2.destroyAllWindows()
