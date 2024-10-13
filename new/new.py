import cv2
from torchvision import transforms
from PIL import Image
import torch
import kagglehub
# Download the model
path = kagglehub.model_download("guru001/mivolo_imdb/pyTorch/imdb_cross_person_4.22_99.46")
print("Path to model files:", path)


# Assuming the model is saved as a .pt file
model_path = path + "/model.pt"

# Load the model
model = torch.load(model_path)
model.eval()  # Set the model to evaluation mode

torch.save(model.state_dict(), "saved_model.pth")
# Define a transformation for the input data (adjust according to your model requirements)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Example transformation
    transforms.ToTensor(),
])

# Initialize video capture
webcam = cv2.VideoCapture(0)

while webcam.isOpened():
    ret, frame = webcam.read()
    if not ret:
        break

    # Convert the frame to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Apply transformations
    input_tensor = transform(pil_image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)

    # Process the output (example: get predicted class)
    predicted_class = output.argmax(dim=1).item()
    confidence = torch.nn.functional.softmax(output, dim=1).max().item()

    # Display results on the frame
    cv2.putText(frame, f"Class: {predicted_class} (Confidence: {confidence:.2f})", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Real-time Inference", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()