from PIL import Image, ImageOps
import os

def resize(input_directory, output_directory, width, height):
    os.makedirs(output_directory, exist_ok=True)

    for item in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, item)):
            img = Image.open(os.path.join(input_directory, item))
            img_width, img_height = img.size
            
            # Calculate the aspect ratio
            aspect_ratio = img_width / img_height
            
            if aspect_ratio > width / height:  # Image is wider than target aspect ratio
                new_height = height
                new_width = int(aspect_ratio * new_height)
            else:  # Image is taller than target aspect ratio
                new_width = width
                new_height = int(new_width / aspect_ratio)

            # Resize image while maintaining aspect ratio
            img = img.resize((new_width, new_height), Image.LANCZOS)

            # Calculate padding
            padding_width = (width - new_width) // 2
            padding_height = (height - new_height) // 2
            
            # Add padding to the image
            img = ImageOps.expand(img, border=(padding_width, padding_height), fill='white')  # Use 'white' for padding fill color

            new_file_name = 'resized-' + item
            img.save(os.path.join(output_directory, new_file_name))

input_directory = r"C:\Users\navon\OneDrive\Documents\ai workd\dataset\test\Cauliflower"
output_directory = r"C:\Users\navon\OneDrive\Documents\ai workd\data"
resize(input_directory, output_directory, 244, 244)
