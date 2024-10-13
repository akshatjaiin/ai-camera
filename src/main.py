# import the opencv library 
import cv2 
import numpy as np

face_detector= cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
gender_model = cv2.dnn.readNetFromCaffe('./models/gender.prototxt', './models/gender_net.caffemodel')
gender_labels = ["Male","Female"]

video =cv2.VideoCapture(0) 

def detectGender(frame):
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print("\nGray",grayImg ,"\n frame", frame)
    faces = face_detector.detectMultiScale(grayImg , scaleFactor=1.2, minNeighbors=5, minSize=(25, 25));
    for (x,y,w,h) in faces:
        face = frame[y:y+h, x:x+w]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (104.0, 177.0, 123.0), swapRB=False)
        # Pass the blob through the gender classification model
        gender_model.setInput(blob)
        gender_pred = gender_model.forward()
        print(gender_pred)
        # Get the predicted gender label
        gender_label = gender_labels[np.argmax(gender_pred[0])]
        # Draw rectangles around the detected faces with the predicted gender label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'{gender_label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)


while(True): 
    # Capture the video frame 
    # by frame 
    ret, frame = video.read() 
    if not ret:
        print("Unable to read Camera");
        break;
    detectGender(frame);
    # Display the resulting frame 
    cv2.imshow('frame', frame) 
      
    if cv2.waitKey(1) & 0xFF == ord('q'): break
  
# After the loop release the cap object 
video.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 

