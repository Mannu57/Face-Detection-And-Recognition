import cv2
import numpy as np

# Load the face detection model
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create an LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Check if the model exists and can be read
model_path = 'trainer/trainer.yml'

try:
    recognizer.read(model_path)  # Try to read the trained model
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Unable to load model: {e}")
    exit(1)

# Initialize webcam
cam = cv2.VideoCapture(0)

frame_count = 0  # Counter to limit the number of frames
max_frames = 1000  # Set the number of frames to process before exiting

while True:
    ret, img = cam.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        id, confidence = recognizer.predict(roi_gray)
        
        # Add logic to handle prediction result
        print(f"ID: {id}, Confidence: {confidence}")
        # Draw rectangle and display ID
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, str(id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Face Recognition", img)

    frame_count += 1
    if frame_count >= max_frames:
        print(f"Processed {max_frames} frames. Exiting...")
        break

    # Break loop on 'ESC' key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
