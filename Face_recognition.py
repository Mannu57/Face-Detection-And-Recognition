import cv2
import numpy as np
import os
import pandas as pd  # Ensure pandas is imported
from datetime import datetime

# Load the trained recognizer and face detection model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')  # Load the trained model
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if the student list exists and read it
student_list_path = "student_list.csv"
if os.path.exists(student_list_path):
    df = pd.read_csv(student_list_path)
    # Strip any extra spaces in column names
    df.columns = df.columns.str.strip()
    
    # Check if 'ID' and 'Name' columns exist
    if 'ID' not in df.columns or 'Name' not in df.columns:
        print("[ERROR] The CSV file must contain 'ID' and 'Name' columns.")
        exit()
else:
    # Create an empty dataframe if the file doesn't exist
    df = pd.DataFrame(columns=["ID", "Name"])
    print("[INFO] student_list.csv not found. Using an empty student list.")

# Start the webcam
cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

# Tracking last recognized face and time
last_id = None
last_recognition_time = None
recognition_timeout = 5  # Time in seconds to reset face recognition

print("[INFO] Recognizing faces. Press ESC to exit.")

while True:
    ret, img = cam.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        id, conf = recognizer.predict(gray[y:y + h, x:x + w])
        
        # Add a confidence threshold to identify faces reliably
        if conf < 60:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            name = df.loc[df['ID'] == id, 'Name'].values[0] if id in df['ID'].values else f"ID {id}"
            label = f"{name} ({id})"
            
            # Reset recognition after the timeout to avoid infinite recognition
            if last_id == id and last_recognition_time:
                if (datetime.now() - last_recognition_time).total_seconds() < recognition_timeout:
                    # Continue showing the previous ID
                    label = f"Last Recognized: {name} ({id})"
                else:
                    last_id = None  # Reset after timeout
            else:
                last_id = id
                last_recognition_time = datetime.now()
        else:
            label = "Unknown"

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 10), font, 0.8, (255, 255, 255), 2)

    cv2.imshow('camera', img)
    
    # Break the loop when 'ESC' key is pressed
    if cv2.waitKey(10) & 0xFF == 27:
        break

# Release the webcam and close all windows
cam.release()
cv2.destroyAllWindows()

