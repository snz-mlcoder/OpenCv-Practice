
import cv2
import numpy as np

# Callback function to detect which quadrant was clicked
def detect_quadrant(event, x, y, flags, param):
    global img
    if event == cv2.EVENT_LBUTTONDOWN:
        if x > width // 2:
            if y > height // 2:
                point_top_left = (width // 2, height // 2)
                point_bottom_right = (width - 1, height - 1)
            else:
                point_top_left = (width // 2, 0)
                point_bottom_right = (width - 1, height // 2)
        else:
            if y > height // 2:
                point_top_left = (0, height // 2)
                point_bottom_right = (width // 2, height - 1)
            else:
                point_top_left = (0, 0)
                point_bottom_right = (width // 2, height // 2)

        # Clear the image (white background)
        img[:] = 255

        # Draw a green rectangle over the selected quadrant
        cv2.rectangle(img, point_top_left, point_bottom_right, (0, 180, 0), -1)

# Main script
if __name__ == '__main__':
    width, height = 640, 480
    img = 255 * np.ones((height, width, 3), dtype=np.uint8)  # white canvas

    cv2.namedWindow('Input window')
    cv2.setMouseCallback('Input window', detect_quadrant)

    while True:
        cv2.imshow('Input window', img)
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()
