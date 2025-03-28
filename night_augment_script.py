import cv2
import numpy as np
import os
import random
import shutil
from tqdm import tqdm # Optional: for a nice progress bar (install with: pip install tqdm)

# --- Configuration ---
INPUT_IMAGE_DIR = './sharp/day/images' # <<< CHANGE THIS to your image folder
INPUT_LABEL_DIR = './sharp/day/labels' # <<< CHANGE THIS to your label folder
OUTPUT_DIR = 'augmented_night'         # <<< Name for the output base directory

# Augmentation Parameters (adjust these ranges as needed)
# Lower brightness factor means darker image
BRIGHTNESS_FACTOR_RANGE = (0.10, 0.25)
# Strength of the blue tint (0-255)
BLUE_TINT_STRENGTH_RANGE = (80, 120)
# How much the blue tint affects the final image (0.0 - 1.0)
BLUE_TINT_WEIGHT_RANGE = (0.1, 0.2)
# --- End Configuration ---

def create_night_effect(image, brightness_factor, blue_strength, blue_weight):
    """Applies a simulated night effect to an image."""
    if image is None:
        print("Warning: Received None image in create_night_effect")
        return None

    try:
        # 1. Reduce Brightness using HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Decrease Value channel (brightness) - ensure it stays within uint8 range
        v_new = np.clip(v * brightness_factor, 0, 255).astype(hsv.dtype)

        hsv_new = cv2.merge([h, s, v_new])
        bright_reduced_image = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

        # 2. Add Blue Tint Overlay
        # Create a solid blue color image of the same size
        blue_overlay = np.zeros_like(bright_reduced_image)
        blue_overlay[:, :, 0] = blue_strength # Set the Blue channel (BGR order)

        # Blend the brightness-reduced image with the blue overlay
        # Ensure image_weight doesn't go below zero if blue_weight is high
        image_weight = max(0.0, 1.0 - blue_weight)
        night_image = cv2.addWeighted(bright_reduced_image, image_weight, blue_overlay, blue_weight, 0)

        # # Optional: Add subtle Gaussian Noise to simulate low-light sensor noise
        # noise_sigma = random.uniform(3, 8) # Adjust sigma for noise level
        # noise = np.random.normal(0, noise_sigma, night_image.shape).astype(np.float32)
        # noisy_image = cv2.add(night_image.astype(np.float32), noise)
        # noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        # return noisy_image

        return night_image

    except Exception as e:
        print(f"Error applying night effect: {e}")
        return None # Return None if any error occurs during processing

def process_dataset(img_dir, lbl_dir, out_base_dir):
    """Processes the dataset to create night-augmented versions."""

    out_img_dir = os.path.join(out_base_dir, 'images')
    out_lbl_dir = os.path.join(out_base_dir, 'labels')

    # --- Input Validation ---
    if not os.path.isdir(img_dir):
        print(f"Error: Input image directory not found: {img_dir}")
        return
    if not os.path.isdir(lbl_dir):
        print(f"Error: Input label directory not found: {lbl_dir}")
        return

    # --- Create Output Directories ---
    try:
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)
        print(f"Output directories created/ensured at: {out_base_dir}")
    except OSError as e:
        print(f"Error creating output directories: {e}")
        return

    # --- List Image Files ---
    try:
        image_files = [f for f in os.listdir(img_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        print(f"Found {len(image_files)} images in {img_dir}")
        if not image_files:
            print("No image files found to process.")
            return
    except OSError as e:
        print(f"Error reading input image directory: {e}")
        return


    # --- Process Images ---
    skipped_count = 0
    error_count = 0
    processed_count = 0

    # Use tqdm for progress bar if available
    file_iterator = tqdm(image_files, desc="Augmenting Images") if 'tqdm' in globals() else image_files

    for img_filename in file_iterator:
        img_path = os.path.join(img_dir, img_filename)
        base_name, _ = os.path.splitext(img_filename)
        label_filename = base_name + '.txt'
        label_path = os.path.join(lbl_dir, label_filename)

        out_img_path = os.path.join(out_img_dir, img_filename) # Keep original filename
        out_label_path = os.path.join(out_lbl_dir, label_filename)

        # Check if corresponding label file exists
        if not os.path.exists(label_path):
            # print(f"Warning: Label file not found for {img_filename}, skipping.") # Reduce console noise
            skipped_count += 1
            continue

        try:
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read image {img_filename}, skipping.")
                skipped_count += 1
                continue

            # Get random parameters for this specific image
            brightness = random.uniform(BRIGHTNESS_FACTOR_RANGE[0], BRIGHTNESS_FACTOR_RANGE[1])
            blue_strength = random.randint(BLUE_TINT_STRENGTH_RANGE[0], BLUE_TINT_STRENGTH_RANGE[1])
            blue_weight = random.uniform(BLUE_TINT_WEIGHT_RANGE[0], BLUE_TINT_WEIGHT_RANGE[1])

            # Apply night effect
            night_image = create_night_effect(image, brightness, blue_strength, blue_weight)

            if night_image is not None:
                # Save augmented image
                success = cv2.imwrite(out_img_path, night_image)
                if not success:
                     print(f"Warning: Failed to write augmented image {out_img_path}")
                     error_count += 1
                     continue # Don't copy label if image failed to save

                # Copy corresponding label file (important: labels don't change for this type of augmentation)
                shutil.copy2(label_path, out_label_path) # copy2 preserves metadata like modification time
                processed_count += 1
            else:
                 print(f"Warning: Failed to apply night effect to {img_filename}, skipping.")
                 error_count += 1


        except Exception as e:
            print(f"Error processing {img_filename}: {e}")
            error_count += 1

    # --- Final Summary ---
    print("\n--- Processing Summary ---")
    print(f"Total images found: {len(image_files)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (missing labels or unreadable images): {skipped_count}")
    print(f"Errors during processing: {error_count}")
    print(f"Augmented images saved to: {out_img_dir}")
    print(f"Corresponding labels copied to: {out_lbl_dir}")
    print("--------------------------")


if __name__ == "__main__":
    # Make sure to update the paths in the Configuration section above
    if INPUT_IMAGE_DIR == 'path/to/your/daytime/images' or \
       INPUT_LABEL_DIR == 'path/to/your/daytime/labels':
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! Please update INPUT_IMAGE_DIR and INPUT_LABEL_DIR     !!!")
        print("!!! in the script before running.                         !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        process_dataset(INPUT_IMAGE_DIR, INPUT_LABEL_DIR, OUTPUT_DIR)