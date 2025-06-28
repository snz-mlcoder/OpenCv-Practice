import cv2
import os
# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

scaling_factor = 0.5  # Scale down for performance
save_counter = 0

# Folder to save faces
os.makedirs("faces", exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles
    for (x, y, w, h) in face_rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save faces on 's' key press
        if cv2.waitKey(1) & 0xFF == ord('s'):
            face_img = frame[y:y+h, x:x+w]
            face_path = f"faces/face_{save_counter}.png"
            cv2.imwrite(face_path, face_img)
            save_counter += 1

    # Add info text
    info_text = f"Faces: {len(face_rects)} | Press 's' to save, ESC to exit"

    cv2.putText(frame, info_text, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Face Detector', frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
