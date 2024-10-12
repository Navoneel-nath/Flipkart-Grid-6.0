import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('best_model.h5')

class_labels = ["Apple", "Banana", "Orange", "Grapes"] 
quality_scores = [0.8, 0.9, 0.7, 0.6]  

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    frame = cv2.resize(frame, (150, 150))  
    frame = frame / 255.0  

    frame = np.expand_dims(frame, axis=0)
    
    predictions = model.predict(frame)
    
    class_index = np.argmax(predictions[0])

    class_label = class_labels[class_index]
    quality_score = quality_scores[class_index]

    cv2.putText(frame, f"Class: {class_label}, Quality: {quality_score:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Fruit Classifier', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()