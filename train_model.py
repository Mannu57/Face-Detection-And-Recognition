import cv2
import numpy as np
import os

# Create 'trainer' directory if it doesn't exist
if not os.path.exists('trainer'):
    os.makedirs('trainer')

# Load the face detection model
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create an LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Prepare data for training
def prepare_data():
    faces = []
    ids = []
    names = {}  # Dictionary to map ID to name
    for image_path in os.listdir('dataset'):
        if image_path.endswith(".jpg"):
            image = cv2.imread(f"dataset/{image_path}")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

            # Get the face region
            face = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in face:
                faces.append(gray[y:y + h, x:x + w])

                # Extract ID and Name from filename
                filename_parts = image_path.split('.')
                id = int(filename_parts[1])  # Numeric ID from filename
                name = filename_parts[2]  # Name from filename
                
                ids.append(id)
                names[id] = name  # Map the ID to name

    return faces, ids, names

faces, ids, names = prepare_data()

# If faces and ids are not empty, train the recognizer
if len(faces) > 0 and len(ids) > 0:
    print("[INFO] Training the recognizer...")
    recognizer.train(faces, np.array(ids))

    # Save the trained model
    recognizer.save('trainer/trainer.yml')
    print("[INFO] Training complete and model saved.")
else:
    print("[ERROR] No faces or IDs found. Cannot train the model.")
