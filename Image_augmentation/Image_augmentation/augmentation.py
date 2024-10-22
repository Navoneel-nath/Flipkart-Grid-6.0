import os
import Augmentor

# Paths
input_dir = r'C:\Users\ritish-flipkart\Downloads\cauli\cauliwhite'
output_dir = r'C:\Users\ritish-flipkart\Downloads\cauli\output'

# Create output directory if not exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize the Augmentor pipeline
p = Augmentor.Pipeline(input_dir, output_dir)

# Add augmentations
p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.random_contrast(probability=0.5, min_factor=0.7, max_factor=1.3)
p.random_brightness(probability=0.5, min_factor=0.7, max_factor=1.3)
p.shear(probability=0.5, max_shear_left=10, max_shear_right=10)

# Generate 200 images
p.sample(200)

print(f"Augmentation completed. 200 images saved in {output_dir}.")