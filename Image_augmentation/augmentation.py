import cv2
import numpy as np
import os
import random

def horizontal_flip(image):
    """Perform horizontal flip on the image."""
    return cv2.flip(image, 1)

def vertical_flip(image):
    """Perform vertical flip on the image."""
    return cv2.flip(image, 0)

def rotate_image(image, angle):
    """Rotate the image by a given angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def random_brightness_contrast(image):
    """Randomly adjust brightness and contrast of the image."""
    alpha = random.uniform(0.8, 1.2)  # Contrast
    beta = random.randint(-30, 30)     # Brightness
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def augment_image(image):
    """Randomly select an augmentation function and apply it to the image."""
    aug_functions = [horizontal_flip, vertical_flip, 
                     lambda img: rotate_image(img, random.randint(-30, 30)),
                     random_brightness_contrast]
    aug_function = random.choice(aug_functions)
    return aug_function(image)

def augment_and_save_images(input_base_dir, output_base_dir, num_augmentations_per_image=10):
    """Augment images and save them to the output directory."""
    os.makedirs(output_base_dir, exist_ok=True)

    for class_name in os.listdir(input_base_dir):
        class_input_dir = os.path.join(input_base_dir, class_name)
        
        if not os.path.isdir(class_input_dir):
            continue

        class_output_dir = os.path.join(output_base_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        image_files = [file for file in os.listdir(class_input_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]

        for img_file in image_files:
            img_path = os.path.join(class_input_dir, img_file)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Error loading image {img_path}. Skipping.")
                continue

            for i in range(num_augmentations_per_image):
                augmented_image = augment_image(image)
                aug_img_name = f'{os.path.splitext(img_file)[0]}_aug_{i}.jpg'
                cv2.imwrite(os.path.join(class_output_dir, aug_img_name), augmented_image)

    print(f"Augmentation completed for all classes. Check the output base directory: {output_base_dir}")

def reduce_noise(image):
    """Reduce noise in the image using Gaussian blur."""
    return cv2.GaussianBlur(image, (5, 5), 0)

def color_correction(image):
    """Apply CLAHE for color correction."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def preprocess_and_save_images(input_base_dir, output_base_dir, target_size=(224, 224)):
    """Preprocess images and save them to the output directory."""
    os.makedirs(output_base_dir, exist_ok=True)

    for class_name in os.listdir(input_base_dir):
        class_input_dir = os.path.join(input_base_dir, class_name)
        
        if not os.path.isdir(class_input_dir):
            continue

        class_output_dir = os.path.join(output_base_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        image_files = [file for file in os.listdir(class_input_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]

        for img_file in image_files:
            img_path = os.path.join(class_input_dir, img_file)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Error loading image {img_path}. Skipping.")
                continue

            # Resize image
            image_resized = cv2.resize(image, target_size)

            # Reduce noise
            image_denoised = reduce_noise(image_resized)

            # Apply color correction
            image_corrected = color_correction(image_denoised)

            # Normalize pixel values to the range [0, 1]
            image_normalized = image_corrected / 255.0

            # Save the processed image
            processed_img_name = f'{os.path.splitext(img_file)[0]}_processed.jpg'
            cv2.imwrite(os.path.join(class_output_dir, processed_img_name), image_normalized * 255)  # Scale back to 0-255 for saving

    print(f"Preprocessing completed for all classes. Check the output base directory: {output_base_dir}")

# Paths for input and output base directories
input_directory = r"C:\Users\navon\OneDrive\Documents\ai workd\resized_data\test"
output_directory = r"C:\Users\navon\OneDrive\Documents\ai workd\augment_test"
output2_directory = r"C:\Users\navon\OneDrive\Documents\ai workd\processed_images"

# Step 1: Augment images for all classes and save them
augment_and_save_images(input_directory, output_directory, num_augmentations_per_image=20)

# Step 2: Preprocess the augmented images and save them to another directory
preprocess_and_save_images(output_directory, output2_directory)