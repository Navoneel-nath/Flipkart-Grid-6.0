import pandas as pd
import qrcode

# Load the CSV data
csv_file = 'fruits_dummy_data.csv'
data = pd.read_csv(csv_file)

# Iterate through each row and generate a QR code
for index, row in data.iterrows():
    # Combine data fields into a string
    qr_data = f"Serial NO: {row['Serial NO']}\nProduct Name: {row['Product Name']}\nManufacturing Date: {row['Manufacturing Date']}\nExpiry Date: {row['Expiry Date']}"
    
    # Generate QR code
    qr_img = qrcode.make(qr_data)
    
    # Save QR code as an image
    qr_img.save(f"qr_code_{row['Product Name']}.png")

print("QR codes generated successfully!")
