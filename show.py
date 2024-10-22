import torch
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# Load the YOLOv5 model from a local path
yolo_model_path = r"C:\Users\navoneel\YOLO\yolov5\runs\train\exp7\weights\best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path, force_reload=True)

# Load the Keras model for quality assessment
quality_model_path = r"C:\Users\pabit\Desktop\PabitraMaharana\FlipcartGrid60\fruit_quality_model.keras"
quality_model = load_model(quality_model_path)

# Quality check threshold
QUALITY_THRESHOLD = 0.1  # Adjust this threshold based on your requirement

# Function to preprocess the image for the Keras model
def preprocess_image(image, target_size=(224, 224)):  # Ensure this matches your Keras model input size
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to perform quality assessment using the Keras model
def assess_quality(image):
    preprocessed_image = preprocess_image(image)
    quality_score = quality_model.predict(preprocessed_image)
    return quality_score[0][0]  # Assuming the model returns a score in a specific range
# Function to run YOLO detection and quality assessment on camera frames
def detect_and_assess_from_camera():
    # Open the camera feed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    best_quality_score = 0  # Initialize variable to keep track of the best quality score

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert frame from BGR (OpenCV format) to RGB (PIL format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Run YOLO detection on the frame
        results = model(pil_image)

        # Extract detected bounding boxes and move to CPU
        detections = results.xyxy[0].cpu().numpy()  # Add .cpu() to move tensor to CPU

        if len(detections) == 0:
            print("No fruit detected. Quality score: 0%")
            cv2.putText(frame, "No fruit detected. Quality score: 0%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            for detection in detections:
                # Extract bounding box coordinates and classification
                x1, y1, x2, y2, confidence, class_id = detection
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert to integers
                label = f"Detected: {results.names[int(class_id)]}"

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Crop the detected fruit region for quality assessment
                cropped_image = pil_image.crop((x1, y1, x2, y2))

                # Assess the quality of the detected fruit
                quality_score = assess_quality(cropped_image)
                quality_result = "Pass" if quality_score >= QUALITY_THRESHOLD else "Not Pass"

                # Update the best quality score if the current one is higher
                if quality_score > best_quality_score:
                    best_quality_score = quality_score

                # Display the quality score and result on the frame
                cv2.putText(frame, f"Quality score: {quality_score * 100:.2f}%", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f"Result: {quality_result}", (x1, y2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Print the result in the terminal
                print(f"Fruit: {results.names[int(class_id)]}, Quality score: {quality_score * 100:.2f}%, Result: {quality_result}")

        # Display the resulting frame
        cv2.imshow('Fruit Detection and Quality Assessment', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    # Print the best quality score after exiting the loop
    print(f"Best Quality Score: {best_quality_score * 100:.2f}%")

# Run the detection and quality assessment from camera feed
detect_and_assess_from_camera()

