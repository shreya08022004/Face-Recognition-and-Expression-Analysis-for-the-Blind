#Importing modules
import cv2 as cv
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from load import load_model_and_encoder

# Load the model and encoder
model, encoder, EMBEDDED_X, Y = load_model_and_encoder()

# Initialize FaceNet and MTCNN
embedder = FaceNet()
detector = MTCNN()

#Get the embeddings of the face
def get_embedding(face_img):
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    yhat = embedder.embeddings(face_img)
    return yhat[0]

# Initialize webcam
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
s = ""
CONFIDENCE_THRESHOLD = 0.5  # Adjust this value as needed

# Capture the frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    faces = detector.detect_faces(frame_rgb)

    for face in faces:
        x, y, w, h = face['box']
        face_img = frame_rgb[y:y+h, x:x+w]
        face_img = cv.resize(face_img, (160, 160))
        embedding = get_embedding(face_img)
        embedding = np.expand_dims(embedding, axis=0)

        ypred = model.predict(embedding)
        ypred_prob = model.predict_proba(embedding)
        label = encoder.inverse_transform(ypred)[0]
        confidence = ypred_prob[0][ypred[0]]

        if confidence < CONFIDENCE_THRESHOLD:
            label = "Guest"

        # Draw bounding box and label on the frame
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv.putText(frame, f'{label} ({confidence*100:.2f}%)', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        s = label
        print(s)

    # Display the frame
    cv.imshow('Face Recognition', frame)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break
    #cv.waitKey(0)  # Wait indefinitely until a key is pressed

cap.release()
cv.destroyAllWindows()
