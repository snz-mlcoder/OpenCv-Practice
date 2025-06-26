import argparse
import cv2

def argument_parser():
    parser = argparse.ArgumentParser(description="Live color space switching using keyboard")
    return parser

if __name__ == '__main__':
    args = argument_parser().parse_args()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    cur_char = -1
    prev_char = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_AREA)
        output = frame.copy()

        # Read key
        c = cv2.waitKey(1)
        if c == 27:  # ESC
            break
        if c > -1 and c != prev_char:
            cur_char = c
            prev_char = c

        # Apply selected transformation
        if cur_char == ord('g'):
            output = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif cur_char == ord('y'):
            output = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        elif cur_char == ord('h'):
            output = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        elif cur_char == ord('r'):
            output = frame

        # Show key instructions on the frame (only if color image)
        info_text = "Keys: g - Grayscale | y - YUV | h - HSV | r - Original | ESC - Exit"
        if len(output.shape) == 3:  # color
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_color = (0, 0, 0)  # black text
            bg_color = (255, 255, 255)  # white background

            # Calculate size of text
            (text_w, text_h), baseline = cv2.getTextSize(info_text, font, font_scale, thickness)
            x, y = 10, output.shape[0] - 10

            # Draw background rectangle behind the text
            cv2.rectangle(output, (x - 5, y - text_h - 5), (x + text_w + 5, y + baseline + 5), bg_color, -1)

            # Overlay the text itself
            cv2.putText(output, info_text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


        window_name = "webcam     to colorspace switch- Press keys "
        cv2.imshow(window_name, output)

    cap.release()
    cv2.destroyAllWindows()
