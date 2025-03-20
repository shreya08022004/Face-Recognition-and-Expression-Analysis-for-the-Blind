#Importing modules
import cv2 as cv
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from load import load_model_and_encoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pyttsx3

#Intialize text-to-speech engine
engine = pyttsx3.init()

# Load the models and encoders
face_recognition_model, face_encoder, EMBEDDED_X, Y = load_model_and_encoder()
emotion_classifier = load_model(r'C:\Users\Shreya Manchikanti\OneDrive\Documents\Mini-Project_Implementation_Expression_Analysis\model.h5')

# Initialize FaceNet and MTCNN
face_embedder = FaceNet()
face_detector = MTCNN()

# Load Haar Cascade for emotion detection
face_classifier = cv.CascadeClassifier(r'C:\Users\Shreya Manchikanti\OneDrive\Documents\Mini-Project_Implementation_Expression_Analysis\haarcascade_frontalface_default.xml')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

CONFIDENCE_THRESHOLD = 0.5  # Adjust this value as needed

#Generates the embeddings of the face i.e. the numrical format of the face for the neural network input
def get_embedding(face_img):
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    yhat = face_embedder.embeddings(face_img)
    return yhat[0]

# Initialize webcam
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

#Capturing frames using web cam
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break
    
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #Converting the image to RGB from BGR 
    faces = face_detector.detect_faces(frame_rgb)
    
    for face in faces:
        x, y, w, h = face['box']                     #Extracting and resizing the image
        face_img = frame_rgb[y:y+h, x:x+w]
        face_img = cv.resize(face_img, (160, 160))
        
        # Face recognition
        embedding = get_embedding(face_img)
        embedding = np.expand_dims(embedding, axis=0)
        ypred = face_recognition_model.predict(embedding)
        ypred_prob = face_recognition_model.predict_proba(embedding)
        face_label = face_encoder.inverse_transform(ypred)[0]
        face_confidence = ypred_prob[0][ypred[0]]
        
        if face_confidence < CONFIDENCE_THRESHOLD:
            face_label = "Guest"
        
        # Emotion detection
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv.resize(roi_gray, (48, 48), interpolation=cv.INTER_AREA)
        
        if np.sum([roi_gray]) != 0:                                     
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            emotion_prediction = emotion_classifier.predict(roi)[0]
            emotion_label = emotion_labels[emotion_prediction.argmax()]
        else:
            emotion_label = "No Face"
        
        # Store the results in a string
        result_string = f"{face_label}, is standing in front of you with a {emotion_label} face"
        print(result_string)

        #Convert the result string to audio
        engine.say(result_string)
        engine.runAndWait()
        
        # Draw bounding box and labels on the frame
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv.putText(frame, f'{face_label} ({face_confidence*100:.2f}%)', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv.putText(frame, emotion_label, (x, y+h+20), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the frame
    cv.imshow('Face and Emotion Recognition', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
