import cv2
import cvlib as cv
import numpy as np
import mediapipe as mp

def enhance_image(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    hsv = cv2.merge((h, s, v))
    enhanced_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return enhanced_frame

def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def detect_faces_mediapipe(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            faces.append(bbox)
    return faces

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

while webcam.isOpened():
    status, frame = webcam.read()

    if not status:
        break

    frame = enhance_image(frame)
    frame = adjust_gamma(frame, gamma=1.5)

    # Detect faces using Mediapipe
    faces = detect_faces_mediapipe(frame)
    padding = 20

    if len(faces) == 0:
        print("No faces detected")
        cv2.putText(frame, "No faces detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        for (startX, startY, w, h) in faces:
            endX = startX + w
            endY = startY + h

            startX = max(0, startX - padding)
            startY = max(0, startY - padding)
            endX = min(frame.shape[1] - 1, endX + padding)
            endY = min(frame.shape[0] - 1, endY + padding)

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            face_crop = np.copy(frame[startY:endY, startX:endX])

            if face_crop.size > 0:
                try:
                    (label, conf) = cv.detect_gender(face_crop)
                    idx = np.argmax(conf)
                    label = label[idx]
                    label_text = f"{label}: {conf[idx] * 100:.2f}%"
                except Exception as e:
                    print(f"Error during gender detection: {e}")
                    label_text = "Error"
            else:
                label_text = "Face crop is empty"

            cv2.putText(frame, label_text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    cv2.imshow("Real-time gender detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
