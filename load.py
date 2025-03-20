#Importing modules
import joblib
import numpy as np

#Loading the saved trained models to pass to face_recognition.py
def load_model_and_encoder():
    # Load the model and encoder
    model = joblib.load('best_face_recognition_model.pkl')
    encoder = joblib.load('label_encoder.pkl')

    # Load the embedded data (if needed)
    data = np.load('faces_embeddingd_done_4classes.npz')
    EMBEDDED_X = data['arr_0']
    Y = data['arr_1']

    return model, encoder, EMBEDDED_X, Y
