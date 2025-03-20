#Importing the modules
import os
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import cv2 as cv
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Define the face loading class
class FACELOADING:
    #Intializing the required variables
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160,160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()

    #Pre-processing the images accordingly and Extracting the coordinates of the image
    def extract_face(self, filename):
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x, y, w, h = self.detector.detect_faces(img)[0]['box']
        x, y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face_arr = cv.resize(face, self.target_size)
        return face_arr

    #Extracting face images
    def load_faces(self, dir):
        FACES = []
        for im_name in os.listdir(dir):
            try:
                path = os.path.join(dir, im_name)
                single_face = self.extract_face(path)
                FACES.append(single_face)
            except Exception as e:
                pass
        return FACES
    
    #Load face images from the sub-directories
    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = os.path.join(self.directory, sub_dir)
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            print(f"Loaded successfully: {len(labels)}")
            self.X.extend(FACES)
            self.Y.extend(labels)
        return np.asarray(self.X), np.asarray(self.Y)

# Load the dataset
faceloading = FACELOADING("dataset_mtcnn")
X, Y = faceloading.load_classes()

# Initialize FaceNet
embedder = FaceNet()

# Function to get embeddings of the face
def get_embedding(face_img):
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    yhat = embedder.embeddings(face_img)
    return yhat[0]

# Convert faces to embeddings
EMBEDDED_X = []
for img in X:
    EMBEDDED_X.append(get_embedding(img))

EMBEDDED_X = np.asarray(EMBEDDED_X)
np.savez_compressed('faces_embeddingd_done_4classes.npz', EMBEDDED_X, Y)

# Encode labels
encoder = LabelEncoder()
encoder.fit(Y)
Y_encoded = encoder.transform(Y)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y_encoded, shuffle=True, random_state=17)

#OLD MODEL only SVM
# Train SVM model
# model = SVC(kernel='linear', probability=True)
# model.fit(X_train, Y_train)

# # Save the model and encoder
# joblib.dump(model, 'face_recognition_model.pkl')
# joblib.dump(encoder, 'label_encoder.pkl')

#NEW MODEL with tuning using GridSearchCV
# param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['linear', 'rbf']}
# grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5)
# grid_search.fit(X_train, Y_train)

# # Best parameters found by grid search
# best_params = grid_search.best_params_

# # Train SVM model with the best parameters
# best_model = SVC(**best_params, probability=True)
# best_model.fit(X_train, Y_train)

#NEW MODEL with tuning using RandomizedSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['linear', 'rbf']}

# Use RandomizedSearchCV with appropriate parameters
random_search = RandomizedSearchCV(SVC(probability=True), param_grid, n_iter=100, cv=5)  # Adjust n_iter for desired search iterations

# Train the model with randomized hyperparameter search
random_search.fit(X_train, Y_train)

# Access best parameters and model
best_params = random_search.best_params_
best_model = SVC(**best_params, probability=True)
best_model.fit(X_train, Y_train)

Y_train_pred = best_model.predict(X_train)
Y_test_pred = best_model.predict(X_test)

train_accuracy = accuracy_score(Y_train, Y_train_pred)
test_accuracy = accuracy_score(Y_test, Y_test_pred)

# Evaluate the model
train_accuracy1 = best_model.score(X_train, Y_train)
test_accuracy1 = best_model.score(X_test, Y_test)

print("Best Parameters:", best_params)
print("Train Score:", train_accuracy1)
print("Test Score:", test_accuracy1)

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

print("Classification Report (Train):")
print(classification_report(Y_train, Y_train_pred))

print("Classification Report (Test):")
print(classification_report(Y_test, Y_test_pred))

print("Confusion Matrix (Test):")
print(confusion_matrix(Y_test, Y_test_pred))


# Save the best model
joblib.dump(best_model, 'best_face_recognition_model.pkl')

# Save the encoder
joblib.dump(encoder, 'label_encoder.pkl')

#Extra added code
#Cross-Validation to track accuracy
cv_scores =cross_val_score(best_model, X_train, Y_train, cv=5)

#Plotting the training accuracy and cross-validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', linestyle='-', label='Cross-Validation Accuracy')
plt.axhline(y=train_accuracy, color='r', linestyle='--', label='Training Accuracy')
plt.title('Training and Cross-Validation Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()