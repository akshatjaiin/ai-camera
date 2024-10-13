import cv2
import numpy as np
import time
from datetime import datetime
import geocoder
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('yolov8n.pt')  # Use the smallest YOLOv8 model for faster inference

# Get current location
def get_location():
    g = geocoder.ip('me')
    return g.latlng if g.ok else g.city

# Initialize variables
gender_count = {'person': 0}  # YOLOv8 detects 'person' class, not gender
incident_count = 0
incident_locations = []
current_location = get_location()

# Initialize webcam
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduced size for better performance
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

# Process frames
while webcam.isOpened():
    status, frame = webcam.read()
    if not status:
        break

    # Get current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Perform YOLOv8 detection
    results = model(frame, stream=True)  # Use streaming mode for better performance

    # Process results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Get class and confidence
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Update person count
            if model.names[cls] == 'person':
                gender_count['person'] += 1

            # Display class and confidence
            cv2.putText(frame, f"{model.names[cls]}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Check for potential incidents (simplified for demonstration)
    current_hour = datetime.now().hour
    if current_hour >= 19 or current_hour <= 6:
        if gender_count['person'] == 1:
            cv2.putText(frame, "ALERT: Lone Person at Night", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            incident_count += 1
            incident_locations.append(current_location)

    # Display person count and other information
    cv2.putText(frame, f"People Detected: {gender_count['person']}", (10, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Time: {current_time}", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    location_text = f"Location: {current_location}" if isinstance(current_location, str) else f"Location: Lat {current_location[0]}, Lon {current_location[1]}"
    cv2.putText(frame, location_text, (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("Women Safety Analytics", frame)

    # Reset person count for next frame
    gender_count['person'] = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
webcam.release()
cv2.destroyAllWindows()

# Output incident report
if incident_count > 0:
    with open('incident_report.txt', 'w') as f:
        f.write(f"Incident Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Incidents: {incident_count}\n\n")
        for i, location in enumerate(incident_locations):
            location_str = f"Lat {location[0]}, Lon {location[1]}" if isinstance(location, list) else location
            f.write(f"Incident {i+1}: Location - {location_str}\n")
else:
    print("No incidents detected.")

print("Processing complete. Incident report saved as 'incident_report.txt' if incidents occurred.")