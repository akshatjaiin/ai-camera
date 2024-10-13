import cv2
import cvlib as cv
import numpy as np
import mediapipe as mp
import time
import threading

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Track gender count and incidents
gender_count = {'male': 0, 'female': 0}
incident_count = 0

# Placeholder for processed frame
processed_frame = None

def process_frame(frame):
    global processed_frame, gender_count, incident_count

    # Enhance and adjust the frame
    frame = enhance_image(frame)
    frame = adjust_gamma(frame, gamma=1.5)

    # Detect faces
    faces = detect_face_enhanced(frame)
    padding = 20
    gender_count = {'male': 0, 'female': 0}

    if len(faces) > 0:
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

                    # Track gender count
                    gender_count[label] += 1

                except Exception as e:
                    print(f"Error during gender detection: {e}")
                    label_text = "Error"
            else:
                label_text = "Face crop is empty"

            cv2.putText(frame, label_text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    # Check for a lone woman at night
    current_time = time.localtime()
    if current_time.tm_hour >= 19 or current_time.tm_hour <= 6:
        if gender_count['female'] == 1 and gender_count['male'] == 0:
            cv2.putText(frame, "ALERT: Lone Woman at Night", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            incident_count += 1

    # Check for a woman surrounded by men
    if gender_count['female'] > 0 and gender_count['male'] > gender_count['female']:
        cv2.putText(frame, "ALERT: Woman Surrounded by Men", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        incident_count += 1

    # Detect hand gestures
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_results = hands.process(rgb_frame)
    
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Analyze hand gestures for SOS
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]

            index_base = hand_landmarks.landmark[5]
            middle_base = hand_landmarks.landmark[9]
            ring_base = hand_landmarks.landmark[13]
            pinky_base = hand_landmarks.landmark[17]

            fingers_closed = 0

            if thumb_tip.x < hand_landmarks.landmark[3].x:
                fingers_closed += 1
            if index_tip.y > index_base.y:
                fingers_closed += 1
            if middle_tip.y > middle_base.y:
                fingers_closed += 1
            if ring_tip.y > ring_base.y:
                fingers_closed += 1
            if pinky_tip.y > pinky_base.y:
                fingers_closed += 1

            if fingers_closed >= 4:
                cv2.putText(frame, "SOS Gesture Detected", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                incident_count += 1

    # Display gender distribution and hotspot information
    cv2.putText(frame, f"Gender Distribution: Male - {gender_count['male']}, Female - {gender_count['female']}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Incident Count: {incident_count}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Save the processed frame
    processed_frame = frame

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

# Initialize webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

# Start the processing thread
processing_thread = None

while webcam.isOpened():
    status, frame = webcam.read()
    if not status:
        break

    # Show real-time frame
    cv2.imshow("Real-time Feed", frame)

    # Start processing the frame in a separate thread
    if processing_thread is None or not processing_thread.is_alive():
        processing_thread = threading.Thread(target=process_frame, args=(frame,))
        processing_thread.start()

    # If processed frame is ready, overlay the effects
    if processed_frame is not None:
        overlay_frame = processed_frame.copy()
        alpha = 0.6  # Transparency factor for overlay
        frame = cv2.addWeighted(overlay_frame, alpha, frame, 1 - alpha, 0)
        cv2.imshow("Processed Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
