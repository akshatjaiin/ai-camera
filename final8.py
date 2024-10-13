import cv2
import cvlib as cv
import numpy as np
import mediapipe as mp
import time
import geocoder
import requests
from datetime import datetime

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Enhance and adjust the image for better detection
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

# Detect faces in the frame
def detect_face_enhanced(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))  # Adjusted scaleFactor and minSize
    return faces

# Get current location (using reverse geocoding for actual address)
def get_location():
    g = geocoder.ip('me')
    if g.ok and g.latlng:
        lat, lon = g.latlng
        location = get_actual_location(lat, lon)
        return location
    return "Location unavailable"

# Convert latitude and longitude to a human-readable address
def get_actual_location(lat, lon):
    try:
        response = requests.get(f'https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=18&addressdetails=1')
        data = response.json()
        return data.get('display_name', 'Unknown location')
    except Exception as e:
        print(f"Error fetching location: {e}")
        return "Unknown location"

# Track the gender count
gender_count = {'male': 0, 'female': 0}

# Track incidents for hotspot detection
incident_count = 0
incident_locations = []

# Use phone camera stream (replace with your phone IP camera stream URL)
stream_url = "http://192.168.29.200:8080/video"
webcam = cv2.VideoCapture(stream_url)

if not webcam.isOpened():
    print("Could not open camera stream")
    exit()

# Get initial location
current_location = get_location()

while webcam.isOpened():
    status, frame = webcam.read()
    if not status:
        break

    # Enhance and adjust the frame
    frame = enhance_image(frame)
    frame = adjust_gamma(frame, gamma=1.5)

    # Flip the frame horizontally to swap left and right
    frame = cv2.flip(frame, 1)

    # Get current time and location
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Detect faces
    faces = detect_face_enhanced(frame)
    padding = 20
    gender_count = {'male': 0, 'female': 0}

    if len(faces) == 0:
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

                    # Track gender count
                    gender_count[label] += 1

                except Exception as e:
                    print(f"Error during gender detection: {e}")
                    label_text = "Error"
            else:
                label_text = "Face crop is empty"

            cv2.putText(frame, label_text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Check for a lone woman at night
    current_hour = datetime.now().hour
    if current_hour >= 19 or current_hour <= 6:
        if gender_count['female'] == 1 and gender_count['male'] == 0:
            cv2.putText(frame, "ALERT: Lone Woman at Night", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            incident_count += 1
            incident_locations.append(current_location)

    # Check for a woman surrounded by men
    if gender_count['female'] > 0 and gender_count['male'] > gender_count['female'] * 2:
        cv2.putText(frame, "ALERT: Woman Surrounded by Men", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        incident_count += 1
        incident_locations.append(current_location)

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
                incident_locations.append(current_location)

    # Display gender distribution and hotspot information
    cv2.putText(frame, f"Gender Distribution: Male - {gender_count['male']}, Female - {gender_count['female']}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Incident Count: {incident_count}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display current time and location
    cv2.putText(frame, f"Time: {current_time}", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Location: {current_location}", (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the video frame
    cv2.imshow("Women Safety Monitoring", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
