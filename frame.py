from PIL import Image, ImageOps
import os

def resize(input_directory, output_directory, width, height):
    os.makedirs(output_directory, exist_ok=True)

    for item in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, item)):
            img = Image.open(os.path.join(input_directory, item))
            img_width, img_height = img.size
            new_width, new_height = width, height
            padding_width = (new_width - img_width) // 2
            padding_height = (new_height - img_height) // 2

            img = ImageOps.expand(img, border=(padding_width, padding_height, padding_width, padding_height), fill=img.getpixel((0, 0)))

            img = img.resize((width, height))

            new_file_name = 'resized-' + item
            img.save(os.path.join(output_directory, new_file_name))

input_directory = r"C:\Users\navon\OneDrive\Documents\ai workd\dataset\test\apple"
output_directory = r"C:\Users\navon\OneDrive\Documents\ai workd\data"
resize(input_directory, output_directory, 3400, 3400)