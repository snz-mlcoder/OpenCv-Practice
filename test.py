import cv2

def nothing(x):
    print(f"Trackbar value changed to: {x}")

cv2.namedWindow('My Window')
cv2.createTrackbar('MySlider', 'My Window', 50, 100, nothing)

while True:
    img = cv2.imread('img/07_flower.png')
    val = cv2.getTrackbarPos('MySlider', 'My Window')
    cv2.putText(img, f'Value: {val}', (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('My Window', img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cv2.destroyAllWindows()
