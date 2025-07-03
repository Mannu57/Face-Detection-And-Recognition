import cv2
import os

# Load the face detection model
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a directory to store the face images if it doesn't exist
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Initialize the webcam
cam = cv2.VideoCapture(0)

# Input your ID (numeric) and Name for identification
face_id = input('Enter your ID (numeric): ')  # Numeric ID
face_name = input('Enter your Name: ')  # Name (string)

print("[INFO] Collecting faces. Look at the camera and wait for the collection process to finish.")

count = 0
while True:
    # Capture frame-by-frame
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Save the captured image
        count += 1
        cv2.imwrite(f"dataset/User.{face_id}.{face_name}.{count}.jpg", gray[y:y+h, x:x+w])  # Save face region only

        # Draw rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, f"ID: {face_id}, Name: {face_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the image
    cv2.imshow('Face Capture', img)

    # Break after 20 images are collected
    if count >= 20:
        break

    # Press 'ESC' to quit the collection process earlier
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the webcam and close all windows
print("[INFO] Face collection complete.")
cam.release()
cv2.destroyAllWindows()
