from flask import Flask, Response, render_template, jsonify
import cv2
from pyfirmata import ArduinoMega, util
import easyocr
from pyzbar import pyzbar
import threading
import time
import os
import glob

# Define a list of known brands
brands = ['Lays', 'Kurkure', 'DairyMilk', 'CocaCola', 'Fanta', 'Sprite', 'Mountain Dew']

# Arduino setup for IR sensor
IR_SENSOR_PIN = 2
board = ArduinoMega('COM5')

it = util.Iterator(board)
it.start()

ir_sensor = board.get_pin(f'd:{IR_SENSOR_PIN}:i')

object_detected = False
object_count = 0
pause_ir_detection = threading.Event()  # Event to pause/resume IR sensor detection
ocr_result = ""
qr_result = ""

# Ensure the OCR_Photos directory exists
OCR_PHOTOS_DIR = 'OCR_Photos'
os.makedirs(OCR_PHOTOS_DIR, exist_ok=True)

# Flask app setup
app = Flask(__name__)

# OCR function
def ocr_scan(image_path):
    global ocr_result
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Increase contrast (optional)
    enhanced_image = cv2.equalizeHist(gray_image)

    # Create an OCR reader object
    reader = easyocr.Reader(['en'])

    # Read text from the preprocessed image
    result = reader.readtext(enhanced_image)

    detected_brands = []
    full_text = ""  # String to accumulate detected text
    for detection in result:
        detected_text = detection[1]
        full_text += detected_text  # Concatenate without spaces

    # Check if any brand is a substring of the concatenated text
    for brand in brands:
        if brand.lower() in full_text.lower():
            detected_brands.append(brand)

    if detected_brands:
        ocr_result = f"Detected brand(s): {', '.join(detected_brands)}"
    else:
        ocr_result = "No brand detected"

    # Save the processed image for later viewing
    processed_image_path = os.path.join(OCR_PHOTOS_DIR, 'processed_image.jpg')
    cv2.imwrite(processed_image_path, enhanced_image)  # Save the processed image
    print(f'Saved processed image to {processed_image_path}')
    
    # Save the full detected text as well, if needed
    text_output_path = os.path.join(OCR_PHOTOS_DIR, 'detected_text.txt')
    with open(text_output_path, 'w') as text_file:
        text_file.write(full_text)
    print(f'Saved detected text to {text_output_path}')


def decode_qr_from_frame(frame):
    global qr_result

    qr_codes = pyzbar.decode(frame)

    for qr_code in qr_codes:
        qr_type = qr_code.type
        if qr_type == 'PDF417':  # Skip PDF417 barcodes
            continue

        qr_data = qr_code.data.decode('utf-8')
        qr_result = f"Decoded QR code: {qr_data} (Type: {qr_type})"
        return qr_result


# Function to save a photo to the OCR_Photos directory
def save_photo(frame, count):
    filename = os.path.join(OCR_PHOTOS_DIR, f'photo_{count}.jpg')
    cv2.imwrite(filename, frame)
    print(f'Saved photo to {filename}')

# IR Sensor function (runs continuously)
def ir_sensor_detection(cap):
    global object_detected, object_count
    try:
        while True:
            pause_ir_detection.wait()  # Pause IR detection when necessary
            sensor_value = ir_sensor.read()
            
            if sensor_value is not None:
                if sensor_value == 1:
                    object_detected = False  
                elif sensor_value == 0 and not object_detected:
                    object_count += 1
                    object_detected = True  

                    # Capture and save the photo without stopping the feed
                    ret, frame = cap.read()
                    if ret:
                        save_photo(frame, object_count)

                    pause_ir_detection.clear()  # Pause IR detection after object is detected
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Exiting IR Sensor detection")
    finally:
        board.exit()

# Resume IR detection after OCR and QR tasks are completed
def resume_ir_detection():
    global object_detected
    object_detected = False
    pause_ir_detection.set()  # Resume IR detection

# Flask route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    # Use IP Webcam URL instead of the default camera
    url = "http://100.66.63.138:8080/video"
    cap = cv2.VideoCapture(url)

    # Start the IR sensor detection with the camera feed
    ir_thread = threading.Thread(target=ir_sensor_detection, args=(cap,))
    ir_thread.daemon = True  # Daemonize thread to allow program exit
    ir_thread.start()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # QR code detection within the frame
        decode_qr_from_frame(frame)

        # Encode frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Create a frame boundary and yield the frame data
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


# Flask route to display the main page
@app.route('/')
def index():
    return render_template('Test.html')

# Flask route to return the object count
@app.route('/get_count')
def get_count():
    return jsonify({'object_count': object_count})

# Flask route to return OCR result
@app.route('/get_ocr_result')
def get_ocr_result():
    return jsonify({'ocr_result': ocr_result})

# Flask route to return QR result
@app.route('/get_qr_result')
def get_qr_result():
    return jsonify({'qr_result': qr_result})

# Thread to handle OCR tasks
def sensor_task():
    while True:
        if object_detected:
            print("Pausing IR sensor detection for OCR scanning...")

            # Get the latest image from the OCR_Photos directory
            list_of_files = glob.glob(os.path.join(OCR_PHOTOS_DIR, '*.jpg'))  # Get all jpg files
            if list_of_files:
                latest_file = max(list_of_files, key=os.path.getctime)  # Get the latest file

                # Run OCR scanning on the latest file
                ocr_scan(latest_file)

            # Resume IR detection after scanning
            print("Resuming IR sensor detection...")
            resume_ir_detection()

        time.sleep(0.1)

# Main function to run Flask app and background tasks
if __name__ == "__main__":
    # Start the IR sensor detection in a separate thread
    pause_ir_detection.set()  # Start IR detection by setting the event

    # Start the sensor task (OCR) in a separate thread
    sensor_thread = threading.Thread(target=sensor_task)
    sensor_thread.daemon = True
    sensor_thread.start()

    # Run Flask app
    app.run(debug=True, use_reloader=False)