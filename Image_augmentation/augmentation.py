import albumentations as A
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Albumentations augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7),
    A.RGBShift(p=0.5)
])

def augment_and_save_images(input_dir, output_dir, num_augmentations_per_image=10):
    """Augments images and saves them to the output directory."""
    os.makedirs(output_dir, exist_ok=True)

    # List of image files
    image_files = [file for file in os.listdir(input_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Select the first 10 images (or fewer if not enough images are present)
    image_files = image_files[:10]

    # Augment images
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        image = cv2.imread(img_path)

        for i in range(num_augmentations_per_image):
            augmented = transform(image=image)['image']

            # Save augmented image
            aug_img_name = f'{os.path.splitext(img_file)[0]}_aug_{i}.jpg'
            cv2.imwrite(os.path.join(output_dir, aug_img_name), augmented)

    print(f"Augmentation completed. Check the output directory: {output_dir}")

# Paths for input and output directories
input_directory = "C:/Users/navon/OneDrive/Documents/img_test/input/class_1"
output_directory = "C:/Users/navon/OneDrive/Documents/img_test/output/class_1"

# Step 1: Augment images and save them
augment_and_save_images(input_directory, output_directory, num_augmentations_per_image=10)

# Verify augmented images are saved
def verify_image_files(directory):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if not files:
        print(f"No images found in {directory}")
    else:
        print(f"Found {len(files)} images in {directory}")

verify_image_files(output_directory)

# Step 2: Load augmented images using Keras ImageDataGenerator
datagen = ImageDataGenerator(rescale=1.0/255.0)  # Normalize pixel values

train_generator = datagen.flow_from_directory(
    output_directory,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # Adjust to 'binary' if you have only two classes
    shuffle=True
)

# Step 3: Define a simple CNN model in Keras
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')  # Number of classes
])

# Step 4: Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train the model
model.fit(train_generator, epochs=10)

# Save the model after training if needed
model.save('trained_model.h5')
