import os

# Set the folder where your images are
image_folder = "../Desktop/nsamples"
label_folder = "../Desktop/nlabels"  # Can be same as image_folder

# Create the label folder if it doesn't exist
os.makedirs(label_folder, exist_ok=True)

# Get list of all .jpg images
image_files = [
    f for f in os.listdir(image_folder)
    if f.lower().endswith('.jpg')
]

# Create empty .txt file for each image
for image_file in image_files:
    base_name = os.path.splitext(image_file)[0]
    label_file = os.path.join(label_folder, f"{base_name}.txt")
    
    # Only create if it doesn't already exist
    if not os.path.exists(label_file):
        open(label_file, 'w').close()

print("Empty label files created successfully.")
