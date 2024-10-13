# Women Safety Analytics System (Working on cv2 and Under Construction with yolo v8)
working explanation: <https://drive.google.com/drive/folders/1--lNfM2sqqslaVNeU06Jt9POdEKjTFxu>
This project is in progress and aims to create a robust, real-time surveillance system focused on improving women’s safety. Using a combination of facial recognition, gender detection, hand gesture analysis, and environmental awareness, the system is designed to raise alerts in potentially unsafe situations, such as detecting SOS gestures or tracking if a woman is surrounded by too many men.
few pictures..

![WhatsApp Image 2024-10-13 at 5 54 18 PM](https://github.com/user-attachments/assets/5b6ce61a-c524-48a9-8045-5066a983d4a3)

![IMG_0311](https://github.com/user-attachments/assets/1ab9befc-fe2e-460b-bef0-a610ec8f8222)

Currently, the system uses OpenCV (`cv2`) for frame-by-frame analysis, but future iterations will integrate **YOLOv8** for more efficient and accurate video-based analysis. 

## Features (Implemented So Far)

- **Real-Time Gender Detection**: Current implementation uses **CVLib** for gender detection and tracks the number of males and females.
- **SOS Hand Gesture Detection**: Uses **MediaPipe** to detect SOS signals based on hand gestures, like closed fist detection.
- **Lone Woman Detection**: Raises an alert if a lone woman is detected between 7 PM and 6 AM.
- **Woman Surrounded by Men**: Alerts when the number of men exceeds the number of women by a 2:1 ratio.
- **Incident Logging**: Logs details like time, location, and the alert triggered for further analysis.

## Planned Improvements

- **YOLOv8 Integration**: Currently, the project processes each frame using OpenCV in real-time. However, this is not the most efficient method, especially for video streams. 
  - **Future Vision**: Switch to **YOLOv8** (You Only Look Once, version 8) for video-based analysis. YOLOv8 will allow:
    - Faster and more accurate detection, running inference on video frames instead of analyzing frame-by-frame.
    - Better performance on detecting multiple faces, gestures, and objects in a single pass over the video.
    - Real-time object tracking using advanced algorithms to improve the reliability of gender and gesture detection in dynamic scenarios.

- **Custom Training on Video Mode**: Train a model on video data specifically for women safety scenarios (e.g., women walking alone at night, suspicious activities) using **YOLOv8**.
  - This will enhance accuracy in complex environments, allowing for a more tailored solution in real-world scenarios.

- **Optimized Incident Counter**: Improve the efficiency of incident counting based on multiple factors like gesture detection consistency and facial tracking over longer periods.
  
- **Geolocation and Incident Reporting**: Currently, geolocation data is basic, but future versions will enhance this with GPS-based tracking and more accurate location mapping, ensuring each incident is logged precisely.

## Current Workflow

1. **Face and Gesture Detection**: The system initializes the webcam, processes frames in real-time using OpenCV, and detects faces and hand gestures.
2. **Gender Identification**: **CVLib** is used to detect the gender of individuals in the frame.
3. **SOS Gesture Detection**: **MediaPipe** identifies hand gestures, especially closed fists, as a potential SOS signal.
4. **Alerting Mechanism**: If an incident is detected (e.g., a lone woman at night or an SOS signal), the system raises an alert and logs it along with timestamp and location.
5. **Manual Logging**: All incidents are currently logged in `incident_report.txt` for later review.

### Limitations (Current State)

- The current frame-by-frame processing using OpenCV is less efficient and can be slow when handling multiple frames in a live video stream.
- The gender and face detection model works well for basic conditions but could be improved with custom training on larger datasets tailored for women safety scenarios.
- Incident reporting and logging need improvements in terms of scalability and automation.

## Requirements

Install the necessary dependencies using:

```bash
pip install opencv-python cvlib mediapipe numpy geocoder
```
Additional Requirements for Future Development (YOLOv8 Integration)
For the planned YOLOv8 integration, you'll need to install the ultralytics package for YOLOv8 support:
```
pip install ultralytics
```

How to Run the Current Version
Clone or download the repository.
Run the following command to start the current frame-by-frame detection:
```
python final7.py
```

Press q to quit the application.
Example Output (Frame-by-Frame Mode)
The system will process each frame and log incidents (if detected) into incident_report.txt with details such as:

Time of incident
Gender counts (males vs. females)
Detected hand gestures (if any)
Geolocation data (latitude and longitude)
Contribution
As this project is still under development, contributions and suggestions are welcome! Feel free to submit issues or pull requests for new features, improvements, or bug fixes.

Areas for Contribution:
YOLOv8 integration for faster and more efficient video processing.
Custom model training for improved gender and gesture detection in dynamic environments.
Enhanced incident logging and reporting system with real-time alerts.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Note: This project is a work-in-progress. Some features mentioned in this document are still under development and are planned for future versions.

