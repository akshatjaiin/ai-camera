import cv2
import cvlib as cv
import numpy as np

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

def detect_face_enhanced(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

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

    faces = detect_face_enhanced(frame)
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
