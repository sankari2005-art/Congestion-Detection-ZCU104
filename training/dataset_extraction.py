import zipfile
import os

zip_path = "/content/drive/MyDrive/obj/coco128.zip"        # Path to your zip file
extract_to = "dataset_coco"   # Folder where files will be extracted

# Create output directory if it doesn't exist
os.makedirs(extract_to, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("Dataset extracted successfully!")
