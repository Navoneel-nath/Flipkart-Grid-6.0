import cv2
import matplotlib.pyplot as plt
import pytesseract
from pytesseract import Output

image = cv2.imread(r'C:\Users\navon\OneDrive\Documents\Flipkart-Grid-6.0\mango.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

text = pytesseract.image_to_string(image, output_type=Output.DICT)
print(text)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresholded = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
text = pytesseract.image_to_string(thresholded, output_type=Output.DICT)
print(text)

