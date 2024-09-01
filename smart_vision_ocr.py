import cv2
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt
import csv
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
csv_file_path = r'c:\Users\navon\OneDrive\Documents\Flipkart-Grid-6.0\Flipkart-Grid-6.0\fruits_vegetables_data.csv'

def process_image(image_path, item_name):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image not found or unable to load for {item_name}.")
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    text = pytesseract.image_to_string(thresholded)

    save_to_csv(item_name, text)
    print(f"Data stored for {item_name}:\n{text}\n")

def capture_and_process(item_name):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 's' to capture an image or 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow("Camera Feed", frame)
        key = cv2.waitKey(1)

        if key == ord('s'):
            captured_image = frame
            gray = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)
            _, thresholded = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

            text = pytesseract.image_to_string(thresholded)

            save_to_csv(item_name, text)
            print(f"Data stored for {item_name}:\n{text}\n")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def save_to_csv(item_name, text):
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["Item Name", "Extracted Text"])

        writer.writerow([item_name, text])

process_image(r'c:\Users\navon\OneDrive\Documents\Flipkart-Grid-6.0\Flipkart-Grid-6.0\mango.jpg', "Mango")

capture_and_process("Mango")

print(f"Data has been saved to {csv_file_path}")

r'''
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
'''
