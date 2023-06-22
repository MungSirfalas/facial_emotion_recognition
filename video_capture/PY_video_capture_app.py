from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import cv2
from keras.models import load_model
from keras.utils import img_to_array
from keras.preprocessing import image

# face detection via Haar feature-based cascade classifiers (Viola y Jones, 2001)
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# select emotion classifier
# MODEL_NAME = "1_base_model.h5"
MODEL_NAME = "2_weighted_loss.h5"
# MODEL_NAME = "3_data_augmentation.h5"
# MODEL_NAME = "4_data_augm_weighted_loss.h5"
MODEL_PATH = "D:/UNIR/Master_InteligenciaArtificial/2_Cuatrimestre/TFM/Develop/msc_ai_thesis_fer/video_capture/" + MODEL_NAME
emotion_classifier = load_model(MODEL_PATH)

# map labels to emotions
map_emotions = {0: "angry", 1:"disgust", 2:"fear", 3: "happy",
                4: "neutral", 5: "sad", 6:"surprise"}

# define function for facial detection, facial framing and emotion recognition
def facial_detection(frame):
    """Function to detect face and perform emotion recognition
       Args:
           frame: image captured with a video capture device
       Returns:
           rectangle: rectangle to limit the detected face
           probability: the highest probability after emotion recognition
           emotion: the emotion corresponding to the highest probability after recognition
    """
    # transform the image to gray scale to apply face detection classifier
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # apply classifier to detect all faces within the captured frame
    faces = face_classifier.detectMultiScale(frame_gray, 1.3, 4)

    """fix scenario where no face is detected"""

    # loop to classify each face detected within a given frame
    for (x, y, w, h) in faces:
        # draw a rectangle in a given frame, the start point corresponds to the upper-left corner
        start_point = (x, y)
        end_point = (x + w, y + h)
        rectangle_color = (0, 255, 0)
        rectangle_thick = 2
        rectangle = cv2.rectangle(frame, start_point, end_point, rectangle_color, rectangle_thick)

        # get input image to emotion classifier, the image must have a given input_size
        input_image = frame_gray[y:y+h, x:x+w]
        input_size = (48, 48)
        model_input = cv2.resize(input_image, input_size).astype("float")/255.0
        model_input = img_to_array(model_input)
        model_input = np.expand_dims(model_input, axis=0)

        # model predictions
        prediction = emotion_classifier.predict(model_input)[0]
        label = prediction.argmax()
        probability = np.max(prediction)
        emotion = "{}: {:.0f}%".format(map_emotions[label], 100 * probability)
        print(emotion)

        # add text to frame
        text = emotion
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

# define function to show in real time the captured images
def view_webcam():
    """Function to show in real time the captured images
       Args:
           None
       Returns:
           None
    """
    # define a global variable for captured images with video capture device
    global cap

    # capture/read image frame by frame and return boolean value "ret"
    # is ret is False, this means no frame was captured therefore we skip the loop
    ret, frame = cap.read()
    if not ret:
        raise Exception("Failed to find video capture device")
    
    # transform the image to gray scale to apply face detection classifier
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # emotion classifier
    facial_detection(frame)

    # captured image to app photoimage
    raw_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    raw_image = Image.fromarray(raw_image)
    app_image = ImageTk.PhotoImage(image=raw_image)
    
    # display photoimage in the corresponding app label
    label_camera.photo_image = app_image
    label_camera.configure(image=app_image)

    # repeat the same process after every N_TIME seconds
    N_TIME = 10
    label_camera.after(N_TIME, view_webcam)

# define function to start the video capture device
def start_webcam():
    """Function to start the video capture device
       Args:
           None
       Returns:
           None
    """
    # define a global variable for captured images with video capture device
    global cap 

    # turn on the video capture device (e.g., webcam) where n_cap_device represents the number
    # of the device to be opened in case of having more than one video capture devices
    N_DEVICE = 0
    cap = cv2.VideoCapture(N_DEVICE)
    if not cap.isOpened():
        raise Exception("Error: Webcam cannot be started")
    
    view_webcam()

# create gui, add title and configure window size
cap = None
app = Tk()
app.title("Emotion Recognition via Facial Expressions")
app.configure(width=800, height=500, background="Turquoise", padx=10, pady=10)

# create a label to display the webcam
label_camera = Label(app)
label_camera.grid(row=0, column=0, columnspan=2)

# create a exit button to close the window
exit_button = Button(app, text="Close", font=("Helvetica", 24), fg='black', background="LightGrey", command=app.destroy)
exit_button.grid(row=1, column=0, rowspan=2, columnspan=2, padx=(50, 50), pady=(15, ))

# trigger video capture and emotion recognition functions
app.after(10, start_webcam)

# infinite loop to display app
app.mainloop()