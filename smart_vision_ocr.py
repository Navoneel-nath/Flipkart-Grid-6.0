import cv2
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image_path = r'C:\Users\navon\OneDrive\Documents\Flipkart-Grid-6.0\Flipkart-Grid-6.0\mango.jpg'
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found or unable to load.")
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.show()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresholded = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    plt.imshow(thresholded, cmap='gray')
    plt.title("Thresholded Image")
    plt.show()

    text = pytesseract.image_to_string(thresholded)

    print("Extracted Text:")
    print(text)

    d = pytesseract.image_to_data(thresholded, output_type=Output.DICT)
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        image_rgb = cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.imshow(image_rgb)
    plt.title("Image with OCR Bounding Boxes")
    plt.show()


