import numpy as np
import cv2

#face_haar_cascade = cv2.CascadeClassifier(cv2.)
cap = cv2.VideoCapture(0)

#if not cap.isOpened():
#    cap = cv2.VideoCapture(0)
#if not cap.isOpened():
#    raise IOError("Cannot open webcam")

while True:
    _, frame = cap.read()

    cv2.imshow('detector', frame)

    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows
