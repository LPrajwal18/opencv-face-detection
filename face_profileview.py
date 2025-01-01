import cv2 as cv

# Iselecting the webcam
cap = cv.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load OpenCV's pre-trained Haar Cascade classifiers to face detection
frontal_face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_profileface.xml')

# Check if the classifiers are loaded correctly
if frontal_face_cascade.empty() or profile_face_cascade.empty():
    print("Error: Could not load Haar Cascade classifiers.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert the frame to grayscale (Haar Cascades work on grayscale images)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    flipped_gray = cv.flip(gray, 1)

    # Detect frontal faces in the grayscale image
    frontal_faces = frontal_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(frontal_faces) > 0:
        # Draw rectangles around the detected frontal faces
        for (x, y, w, h) in frontal_faces:
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame = cv.putText(frame, "Frontal Face Detected", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        # Detect left profile faces in the grayscale image
        left_profile_faces = profile_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Detect right profile faces in the flipped grayscale image
        right_profile_faces = profile_face_cascade.detectMultiScale(flipped_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected left profile faces
        for (x, y, w, h) in left_profile_faces:
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            frame = cv.putText(frame, "Left Profile Face Detected", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw rectangles around the detected right profile faces
        for (x, y, w, h) in right_profile_faces:
            x = frame.shape[1] - x - w  # Correct the x-coordinate for the flipped image
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            frame = cv.putText(frame, "Right Profile Face Detected", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame with face detections
    cv.imshow("Frontal and Profile Face Detection", frame)

    # Exit condition (press 'q' to quit)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()