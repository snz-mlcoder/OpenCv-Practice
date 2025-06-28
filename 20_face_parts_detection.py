import cv2

# Load cascades from main folder
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_mcs_mouth.xml')

# Check if cascades loaded correctly
if face_cascade.empty() or eye_cascade.empty() or nose_cascade.empty() or mouth_cascade.empty():
    raise IOError("One or more cascade files failed to load. Please make sure all .xml files are in the same folder as the script.")

# Start webcam
cap = cv2.VideoCapture(0)
scaling_factor = 0.6

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Nose
        nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        for (nx, ny, nw, nh) in nose:
            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 255, 255), 2)
            break  # One nose is enough

        # Mouth
        mouth = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=20)
        for (mx, my, mw, mh) in mouth:
            if my > h // 2:  # avoid eyes being mistaken for mouth
                cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
                break  # One mouth is enough

    cv2.imshow('Live Face Feature Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
