import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import sys
import io

# Fix encoding issues by setting UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load the trained model without the optimizer state
model = load_model('fruit_quality_model_texture_color_shape.keras', compile=False)

# Define the class labels based on your dataset
class_labels = ['apple', 'cucumber', 'garlic', 'ginger', 'green_chili', 
                'guava', 'lemon', 'onion', 'potato', 'tomato']

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (800, 800))  # Resize to the input shape of the model
    img = img.astype('float32') / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict the fruit/vegetable and quality
def predict_image(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Use the model to make predictions
    prediction = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(prediction)]
    
    # Quality calculation based on confidence
    quality_percentage = np.max(prediction) * 100

    # Display the image with prediction
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(f'Predicted: {predicted_class} - Quality: {quality_percentage:.2f}%')
    plt.axis('off')
    plt.show()

    # Adjust the quality threshold for spoilage
    if quality_percentage < 65:  # Adjust this threshold as needed
        quality_status = 'Not Passed'
    else:
        quality_status = 'Passed'

    print(f'The fruit/vegetable is predicted as: {predicted_class}')
    print(f'Quality percentage: {quality_percentage:.2f}%')
    print(f'Quality status: {quality_status}')

# Test the function with your image
image_path = r"C:\Users\navon\OneDrive\Documents\ai workd\img3.jpeg"  # Change to your image path
predict_image(image_path)
