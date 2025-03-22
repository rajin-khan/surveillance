import os
from PIL import Image

# Path to your folder with the negative images
folder_path = "../Desktop/nsamples"

# Supported image formats
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Create output list
image_files = sorted([
    f for f in os.listdir(folder_path)
    if f.lower().endswith(image_extensions)
])

# Process each image
for i, filename in enumerate(image_files, start=1):
    base_name = f"negative_sample{i}"
    output_name = f"{base_name}.jpg"
    src_path = os.path.join(folder_path, filename)
    dst_path = os.path.join(folder_path, output_name)

    # Open image and convert to RGB (required for JPG)
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img.save(dst_path, format="JPEG")

    # Remove original if it wasn't already a .jpg
    if not filename.lower().endswith('.jpg'):
        os.remove(src_path)

print("All images converted to .jpg and renamed.")
