import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys
import io

# Set the encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load the trained model without the optimizer state
model = load_model('best_fruit_model.keras', compile=False)

# Define the class labels based on your dataset
class_labels = ['Apple', 'Cucumber', 'Garlic', 'Ginger', 'Green Chili', 'Tomato']

# Function to read and resize the image to the expected input size
def load_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    img = cv2.resize(img, (224, 224))  # Resize to the input shape of the model
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img.astype('float32') / 255.0  # Normalize the image to [0, 1] range
    return img

# Function to assess quality based on model predictions
def assess_quality(prediction):
    quality_threshold = 0.65  # Adjust based on your results
    predicted_class_index = np.argmax(prediction)
    confidence_score = np.max(prediction)

    if confidence_score < quality_threshold:
        quality_status = 'Not Passed'
    else:
        quality_status = 'Passed'

    return quality_status, predicted_class_index, confidence_score

# Function to predict the fruit/vegetable and quality
def predict_image(frame):
    # Load and resize the image to the expected input size
    loaded_image = load_image(frame)

    # Use the model to make predictions
    prediction = model.predict(loaded_image)

    # Assess the quality of the prediction
    quality_status, predicted_class_index, confidence_score = assess_quality(prediction)

    # Check if the index is valid
    if predicted_class_index < len(class_labels):
        predicted_class = class_labels[predicted_class_index]
    else:
        predicted_class = 'Unknown'

    quality_percentage = confidence_score * 100  # Convert confidence score to percentage

    return predicted_class, quality_percentage, quality_status

# Main function to capture video from the camera and predict in real-time
def main():
    # Open the default camera (0)
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Predict the fruit/vegetable and quality
        predicted_class, quality_percentage, quality_status = predict_image(frame)

        # Display the predictions on the frame
        result_text = f'Best Guess: {predicted_class} | Quality: {quality_percentage:.2f}% | Status: {quality_status}'
        cv2.putText(frame, result_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Show the frame
        cv2.imshow('Vegetable Quality Assessment', frame)

        # Print the best guess in the terminal
        print(f'Best Guess: {predicted_class} | Quality: {quality_percentage:.2f}% | Status: {quality_status}')

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close the window
    cap.release()
    cv2.destroyAllWindows()

# Run the main function
if __name__ == "__main__":
    main()
