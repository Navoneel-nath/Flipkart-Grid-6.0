import cv2
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image_path = r'C:\Users\navon\OneDrive\Documents\Flipkart-Grid-6.0\Flipkart-Grid-6.0\mango.jpg'
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found or unable to load.")
    exit()

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')  
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, thresholded = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

text = pytesseract.image_to_string(thresholded, output_type=Output.DICT)
print("Extracted Text:", text['text'])


