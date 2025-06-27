
import cv2
import numpy as np

# Global variables
drawing = False
x_init, y_init = -1, -1
top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global x_init, y_init, drawing, top_left_pt, bottom_right_pt

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_init, y_init = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        top_left_pt = (min(x_init, x), min(y_init, y))
        bottom_right_pt = (max(x_init, x), max(y_init, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        top_left_pt = (min(x_init, x), min(y_init, y))
        bottom_right_pt = (max(x_init, x), max(y_init, y))

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    cv2.namedWindow('Webcam')
    cv2.setMouseCallback('Webcam', draw_rectangle)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for performance
        img = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        # Apply negative effect to ROI
        (x0, y0), (x1, y1) = top_left_pt, bottom_right_pt
        if x0 >= 0 and y0 >= 0 and x1 > x0 and y1 > y0:
            img[y0:y1, x0:x1] = 255 - img[y0:y1, x0:x1]

        # Show result
        cv2.imshow('Webcam', img)

        key = cv2.waitKey(1)
        if key == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()
