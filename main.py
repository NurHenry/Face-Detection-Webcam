import cv2

# Use absolute path for the XML file to avoid potential issues
face_cascade_path = 'path/to/haarcascade_frontalface_default.xml'
cap = cv2.VideoCapture(0)

# Initialize the face cascade classifier outside the loop for performance
face_cascade = cv2.CascadeClassifier(face_cascade_path)

while True:
    # Read the frame and check if it was successfully read
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Gesichtserkennung', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
