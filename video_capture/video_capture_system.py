import numpy as np
import cv2

# face detection via Haar feature-based cascade classifiers (Viola y Jones, 2001)
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# emotion_classifier = 

# turn on the video capture device (e.g., webcam) where n_cap_device represents the number
# of the device to be opened in case of having more than one video capture devices
n_cap_device = 0
cap = cv2.VideoCapture(n_cap_device)
if not cap.isOpened():
   raise IOError("Error: Webcam cannot be started")

# optional to record and to save captured video
save_video = False
if save_video:
    four_cc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter("output_video.avi", four_cc, fps=20, frameSize=[640, 800])
else:
    pass 

# loop to process frames in real-time captured video
while True:
    # capture/read image frame by frame and return boolean value "ret"
    # is ret is False, this means no frame was captured therefore we skip the loop
    ret, frame = cap.read()
    if not ret:
        continue

    # transform the image to gray scale to apply face detection classifier
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # apply classifier to detect all faces within the captured frame
    faces = face_classifier.detectMultiScale(frame_gray, 1.15, 3)

    # loop to classify each face detected within a given frame
    for (x, y, w, h) in faces:
        # draw a rectangle in a given frame, the start point corresponds to the upper-left corner
        start_point = (x, y)
        end_point = (x + w, y + h)
        rectangle_color = (0, 255, 0)
        rectangle_thick = 2
        cv2.rectangle(frame, start_point, end_point, rectangle_color, rectangle_thick)

        # get input image to emotion classifier, the image must have a given input_size
        model_input = frame_gray[y:y+h, x:x+w]
        input_size = (48, 48)
        # model_input = cv2.resize(model_input, input_size, interpolation=cv2.INTER)

        # model predictions

        # add text to frame
        text = "tristeza"
        text_font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 2
        text_color = (255, 255, 255)
        text_thick = 2
        # line_type = cv2.LINE_AA
        text_size, _ = cv2.getTextSize(text, text_font, font_scale, text_thick)
        text_w, text_h = text_size
        text_position = (x, y)
        bg_start_point = (x - int(rectangle_thick / 2), y - text_h - font_scale - rectangle_thick - 1)
        bg_end_point = (x + text_w, y - rectangle_thick)
        bg_thick = -1
        cv2.rectangle(frame, bg_start_point, bg_end_point, rectangle_color, bg_thick)
        cv2.putText(frame, text, text_position, text_font, font_scale, text_color, text_thick)

    # show the captured frames, its components, and assign a name
    cv2.imshow('Emotion Recognition via Facial Expressions', frame)

    # use "q" key to close the window, the window will display 1 frame per wait_ms milliseconds
    wait_ms = 1
    if cv2.waitKey(wait_ms) & 0xFF==ord("q"):
        break

# turn off video capture device (e.g., webcam)
cap.release()
# close all windows
cv2.destroyAllWindows
