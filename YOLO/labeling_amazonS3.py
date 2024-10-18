import boto3
import os

def detect_labels(photo, bucket):
    session = boto3.Session(profile_name='profile-name')
    client = session.client('rekognition')

    response = client.detect_labels(
        Image={'S3Object': {'Bucket': bucket, 'Name': photo}},
        MaxLabels=10
    )

    output = f'Detected labels for {photo}\n\n'

    for label in response['Labels']:
        output += f"Label: {label['Name']}\n"
        output += f"Confidence: {label['Confidence']}\n"
        output += "Instances:\n"

        for instance in label['Instances']:
            output += " Bounding box\n"
            output += f" Top: {instance['BoundingBox']['Top']}\n"
            output += f" Left: {instance['BoundingBox']['Left']}\n"
            output += f" Width: {instance['BoundingBox']['Width']}\n"
            output += f" Height: {instance['BoundingBox']['Height']}\n"
            output += f" Confidence: {instance['Confidence']}\n\n"

        output += "Parents:\n"
        for parent in label['Parents']:
            output += f" {parent['Name']}\n"

        output += "Aliases:\n"
        for alias in label['Aliases']:
            output += f" {alias['Name']}\n"

        output += "Categories:\n"
        for category in label['Categories']:
            output += f" {category['Name']}\n"
        output += "----------\n\n"

    return output

def process_images_in_directory(directory, bucket):
    output_file = 'labels_output.txt'  # File to save the output

    with open(output_file, 'w') as f:
        for filename in os.listdir(directory):
            if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
                photo_path = filename
                output = detect_labels(photo_path, bucket)
                f.write(output)  # Write the output to the file

    print(f"Label information for all images has been written to {output_file}")

def main():
    directory = 'path/to/your/image/directory'  # Change to your directory path
    bucket = 'your-bucket-name'  # Your S3 bucket name
    process_images_in_directory(directory, bucket)

if __name__ == "__main__":
    main()
