import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize MediaPipe Face Detection and Hands
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Enhance and adjust the image for better detection
def enhance_image(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v = clahe.apply(v)
    hsv = cv2.merge((h, s, v))
    enhanced_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return enhanced_frame

def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Track the gender count
gender_count = {'male': 0, 'female': 0}

# Track incidents for hotspot detection
incident_count = 0
last_incident_time = 0
hand_closed = False

# Initialize webcam
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

while webcam.isOpened():
    status, frame = webcam.read()
    if not status:
        break

    # Flip the frame horizontally to swap left and right
    frame = cv2.flip(frame, 1)

    # Enhance and adjust the frame
    frame = enhance_image(frame)
    frame = adjust_gamma(frame, gamma=1.2)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_results = face_detection.process(rgb_frame)
    gender_count = {'male': 0, 'female': 0}

    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, bbox, (0, 255, 0), 2)

            # For simplicity, we'll use a random gender assignment
            # In a real scenario, you'd use a proper gender classification model
            gender = np.random.choice(['male', 'female'])
            gender_count[gender] += 1

            cv2.putText(frame, f"{gender}", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "No faces detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Check for a lone woman at night
    current_time = time.localtime()
    if current_time.tm_hour >= 19 or current_time.tm_hour <= 6:
        if gender_count['female'] == 1 and gender_count['male'] == 0:
            cv2.putText(frame, "ALERT: Lone Woman at Night", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if time.time() - last_incident_time > 5:  # 5 seconds cooldown
                incident_count += 1
                last_incident_time = time.time()

    # Check for a woman surrounded by men
    if gender_count['female'] > 0 and gender_count['male'] > gender_count['female'] * 2:
        cv2.putText(frame, "ALERT: Woman Surrounded by Men", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if time.time() - last_incident_time > 5:  # 5 seconds cooldown
            incident_count += 1
            last_incident_time = time.time()

    # Detect hand gestures
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
                if not hand_closed:
                    cv2.putText(frame, "SOS Gesture Detected", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if time.time() - last_incident_time > 5:  # 5 seconds cooldown
                        incident_count += 1
                        last_incident_time = time.time()
                    hand_closed = True
            else:
                hand_closed = False

    # Display gender distribution and hotspot information
    cv2.putText(frame, f"Gender Distribution: Male - {gender_count['male']}, Female - {gender_count['female']}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Incident Count: {incident_count}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show the frame
    cv2.imshow("Surveillance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()