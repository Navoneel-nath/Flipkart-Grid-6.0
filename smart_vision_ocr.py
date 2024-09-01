import cv2
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image
image_path = r'C:\Users\navon\OneDrive\Documents\Flipkart-Grid-6.0\Flipkart-Grid-6.0\mango.jpg'
image = cv2.imread(image_path)

# Verify if image is loaded
if image is None:
    print("Error: Image not found or unable to load.")
    exit()

# Display the image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Hide axes
plt.show()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresholded = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Extract text using OCR
text = pytesseract.image_to_string(thresholded, output_type=Output.DICT)
print("Extracted Text:", text['text'])


