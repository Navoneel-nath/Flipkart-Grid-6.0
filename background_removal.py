import cv2
import os

# Use only PNG image

# Import the image
file_name = "C:\\Users\\navon\\OneDrive\\Documents\\robottics and ai\\apple.png"

# Read the image
src = cv2.imread("apple.png")

# Invert the colors of the image
srcl = 255 - src

# Convert the image to grayscale
tmp = cv2.cvtColor(srcl, cv2.COLOR_BGR2GRAY)

# Apply threshold technique
_, alpha = cv2.threshold(tmp, 15, 255, cv2.THRESH_BINARY)

# Split the original image into its red, green, and blue channels
b, g, r = cv2.split(src)

# Create a list of the red, green, blue channels, and alpha channel
rgba = [b, g, r, alpha]

# Merge the channels into a single image
dst = cv2.merge(rgba, 4)

# Write the resulting image to a new file
cv2.imwrite('output.png', dst)

#used os module to use open function 
os.system(f'code {"apple.png"}')