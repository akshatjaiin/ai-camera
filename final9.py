import cv2
import numpy as np
import mediapipe as mp
import geocoder
from datetime import datetime
import os

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=8, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to enhance and adjust the image
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

# Function to get accurate location
def get_location():
    g = geocoder.ip('me')
    if g.ok:
        return f"Lat {g.latlng[0]}, Lon {g.latlng[1]}"
    else:
        return "Location unknown"

# Initialize webcam
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

# Initialize counters
incident_count = 0
incident_locations = []

# Get initial location
current_location = get_location()

while webcam.isOpened():
    status, frame = webcam.read()
    if not status:
        break

    # Enhance and adjust the frame
    frame = enhance_image(frame)
    frame = adjust_gamma(frame, gamma=1.5)
    frame = cv2.flip(frame, 1)

    # Get current time and location
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face detection
    results = face_detection.process(rgb_frame)

    face_count = 0
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, bbox, (0, 255, 0), 2)
            face_count += 1

    # Display face count
    cv2.putText(frame, f"Faces detected: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
                cv2.putText(frame, "SOS Gesture Detected", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                incident_count += 1
                incident_locations.append(current_location)

    # Display information
    cv2.putText(frame, f"Incident Count: {incident_count}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Time: {current_time}", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Location: {current_location}", (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("Women Safety Analytics", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and perform cleanup
webcam.release()
cv2.destroyAllWindows()

# Output the incident data
if incident_count > 0:
    with open('incident_report.txt', 'w') as f:
        f.write(f"Incident Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Incidents: {incident_count}\n\n")
        for i, location in enumerate(incident_locations):
            f.write(f"Incident {i+1}: Location - {location}\n")
else:
    print("No incidents detected.")

print("Processing complete. Incident report saved as 'incident_report.txt' if incidents occurred.")