# Face Detection using OpenCV

## Description

This script uses OpenCV to perform real-time face detection using your webcam. It utilizes pre-trained Haar Cascade classifiers to detect frontal and profile faces. The detected faces are highlighted with rectangles and labeled appropriately.

## Requirements

- Python 3.11
- OpenCV library for Python

## Installation

1. Make sure you have Python installed on your system. You can download it from [python.org](https://www.python.org/).

2. Install the OpenCV library using pip:

    ```bash
    pip install opencv-python
    ```

## Usage

1. Save the script to a file, for example, `face_detection.py`.

2. Run the script using Python:

    ```bash
    python face_detection.py
    ```

3. The script will open your webcam and start detecting faces. You can quit the application by pressing the 'q' key.

## Script Explanation

1. **Webcam Initialization**:
    ```python
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    ```

    This part of the code initializes the webcam. If the webcam cannot be opened, it prints an error message and exits the program.

2. **Loading Haar Cascade Classifiers**:
    ```python
    frontal_face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_profileface.xml')

    if frontal_face_cascade.empty() or profile_face_cascade.empty():
        print("Error: Could not load Haar Cascade classifiers.")
        exit()
    ```

    This part loads the pre-trained Haar Cascade classifiers for detecting frontal and profile faces. If the classifiers cannot be loaded, it prints an error message and exits the program.

3. **Face Detection Loop**:
    ```python
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flipped_gray = cv.flip(gray, 1)

        frontal_faces = frontal_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(frontal_faces) > 0:
            for (x, y, w, h) in frontal_faces:
                frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                frame = cv.putText(frame, "Frontal Face Detected", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            left_profile_faces = profile_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            right_profile_faces = profile_face_cascade.detectMultiScale(flipped_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in left_profile_faces:
                frame = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                frame = cv.putText(frame, "Left Profile Face Detected", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            for (x, y, w, h) in right_profile_faces:
                x = frame.shape[1] - x - w
                frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                frame = cv.putText(frame, "Right Profile Face Detected", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv.imshow("Frontal and Profile Face Detection", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    ```

    This loop continuously captures frames from the webcam, converts them to grayscale, and detects faces using the Haar Cascade classifiers. It draws rectangles around detected faces and labels them. The loop runs until the 'q' key is pressed.

4. **Resource Release**:
    ```python
    cap.release()
    cv.destroyAllWindows()
    ```

    This part releases the webcam and closes all OpenCV windows.

## Additional Notes

- Ensure your webcam is properly connected and accessible.
- Adjust the `scaleFactor` and `minNeighbors` parameters in the `detectMultiScale` method to improve detection accuracy based on your specific use case.
