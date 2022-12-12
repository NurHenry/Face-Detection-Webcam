import cv2

# Create a haar cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start Webcam
cap = cv2.VideoCapture(0)

while True:
    # Read Frames
    _, frame = cap.read()

    # Konvert Frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face Tracking
    faces = face_cascade.detectMultiScale(gray)

    # Create Circel around the Face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Shows the Frame
    cv2.imshow('Dgesichtserkennung', frame)

    # Press Q to end
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
