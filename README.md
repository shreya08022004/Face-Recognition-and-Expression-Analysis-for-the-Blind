# Face Recognition and Expression Analysis for the Blind
- The project consists of two models: Face Recognition Model, Expression Analysis Model
## Face Recognition Model
- The dataset for this model is self-curated which consists of the faces of people.
- The dataset consists of different folders labelled with a person's name.
- Each folder consists of few pictures of that specific person.
- The model is built using MTCNN for face recognition, FaceNet for feature extraction and SVM for face classification.
## Expression Analysis Model
- The dataset for this model is taken from kaggle. The link is provided in the file named "Emotion Analysis Model Dataset".
- Haar-Cascade is used for face detection.
- A 6-layer CNN is built and trained for feature extraction and expression classification.
