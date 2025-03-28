import os
import cv2
import numpy as np
# import argparse # No longer needed
import shutil
from tqdm import tqdm
import math

# --- Helper Functions (apply_gamma, apply_vignette, apply_tint - unchanged) ---

def apply_gamma(image, gamma=1.0):
    """Applies gamma correction to the image."""
    if gamma == 1.0:
        return image
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_vignette(image, strength=0.5, radius_scale=1.5):
    """
    Applies a vignette effect (darker corners).
    Handles float or uint8 input, returns float.
    """
    if strength <= 0:
        return image.astype(np.float32) if image.dtype == np.uint8 else image

    rows, cols = image.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, int(cols * radius_scale))
    kernel_y = cv2.getGaussianKernel(rows, int(rows * radius_scale))
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    mask = mask / mask.max() # Normalize 0-1
    mask = 1.0 - (strength * (1.0 - mask)) # Invert relationship for darkening

    image_float = image.astype(np.float32) if image.dtype == np.uint8 else image

    if len(image.shape) == 2: # Grayscale
        vignetted_image = image_float * mask
    else: # BGR
        mask_3d = np.stack([mask] * 3, axis=-1)
        vignetted_image = image_float * mask_3d

    return vignetted_image # Return float image

def apply_tint(image, tint_color=(0, 255, 0), intensity=0.3):
    """
    Applies a color tint. Assumes input is grayscale uint8.
    Returns BGR uint8 image.
    """
    if intensity <= 0:
         return image # Return original grayscale

    if len(image.shape) == 2:
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
         output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output_image = image.copy()

    if output_image.dtype != np.uint8:
        output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    overlay = np.full(output_image.shape, tint_color, dtype=np.uint8)
    cv2.addWeighted(overlay, intensity, output_image, 1.0 - intensity, 0, output_image)
    return output_image

# --- Main Simulation Function (Unchanged from previous version) ---

def simulate_night_vision(
    image,
    exposure_factor=1.0,
    noise_std_dev=15.0,
    gamma=1.8,
    blur_ksize=3,
    vignette_strength=0.4,
    vignette_radius_scale=1.5,
    use_clahe=False,
    clahe_clip=2.0,
    clahe_grid=8,
    tint_color_str='0,255,0',
    tint_intensity=0.2
    ):
    """
    Simulates night vision effects on an image with more options.
    (Function body remains the same as the previous version)
    """
    if image is None: return None
    if len(image.shape) == 3: gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2: gray_image = image
    else: return None

    processed_image = gray_image.astype(np.float32)
    if 0 < exposure_factor < 1.0: processed_image *= exposure_factor
    elif exposure_factor <= 0: processed_image = np.zeros_like(processed_image, dtype=np.float32)

    processed_image_uint8 = np.clip(processed_image, 0, 255).astype(np.uint8)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
        processed_image_uint8 = clahe.apply(processed_image_uint8)

    if gamma != 1.0:
        processed_image_uint8 = apply_gamma(processed_image_uint8, gamma)

    processed_image_float = processed_image_uint8.astype(np.float32)

    if blur_ksize > 1 and blur_ksize % 2 == 1:
         processed_image_float = cv2.GaussianBlur(processed_image_float, (blur_ksize, blur_ksize), 0)
    elif blur_ksize > 1:
         print(f"Warning: Blur kernel size ({blur_ksize}) must be odd. Skipping blur.")

    if noise_std_dev > 0:
        mean = 0
        noise = np.random.normal(mean, noise_std_dev, processed_image_float.shape)
        processed_image_float += noise

    if vignette_strength > 0:
        processed_image_float = apply_vignette(processed_image_float, vignette_strength, vignette_radius_scale)

    final_image_uint8 = np.clip(processed_image_float, 0, 255).astype(np.uint8)

    if tint_intensity > 0:
        try:
            b, g, r = map(int, tint_color_str.split(','))
            tint_color = (b, g, r)
            final_output_image = apply_tint(final_image_uint8, tint_color, tint_intensity)
        except ValueError:
            print(f"Warning: Invalid tint color format '{tint_color_str}'. Use 'B,G,R'. Skipping tint.")
            final_output_image = final_image_uint8
    else:
        final_output_image = final_image_uint8

    if final_output_image.dtype != np.uint8:
         final_output_image = np.clip(final_output_image, 0, 255).astype(np.uint8)

    return final_output_image


# --- Dataset Processing Function (Unchanged - still accepts args dict) ---

def process_dataset(input_folder, output_folder, sim_args):
    """
    Processes the dataset folder, applying enhanced night vision simulation.
    (Function body remains the same as the previous version)
    """
    images_in_dir = os.path.join(input_folder, 'images')
    labels_in_dir = os.path.join(input_folder, 'labels')
    images_out_dir = os.path.join(output_folder, 'images')
    labels_out_dir = os.path.join(output_folder, 'labels')

    if not os.path.isdir(input_folder): print(f"Error: Input folder '{input_folder}' not found."); return
    if not os.path.isdir(images_in_dir): print(f"Error: Input 'images' subfolder not found in '{input_folder}'."); return
    if not os.path.isdir(labels_in_dir): print(f"Error: Input 'labels' subfolder not found in '{input_folder}'."); return

    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)
    print(f"Output will be saved to: {output_folder}")
    print(f"Simulation parameters: { {k: sim_args[k] for k in sorted(sim_args)} }")

    image_files = [f for f in os.listdir(images_in_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

    if not image_files: print(f"No image files found in {images_in_dir}."); return
    print(f"Found {len(image_files)} images to process.")

    processed_count = 0
    skipped_count = 0
    for img_filename in tqdm(image_files, desc="Processing Images"):
        img_in_path = os.path.join(images_in_dir, img_filename)
        base_name = os.path.splitext(img_filename)[0]
        label_filename = base_name + '.txt'
        label_in_path = os.path.join(labels_in_dir, label_filename)
        img_out_path = os.path.join(images_out_dir, img_filename)
        label_out_path = os.path.join(labels_out_dir, label_filename)

        if not os.path.exists(label_in_path): skipped_count += 1; continue
        image = cv2.imread(img_in_path)
        if image is None: skipped_count += 1; continue

        try: night_vision_image = simulate_night_vision(image, **sim_args)
        except Exception as e: print(f"\nError processing image '{img_filename}': {e}"); skipped_count += 1; continue
        if night_vision_image is None: skipped_count += 1; continue

        try:
            success = cv2.imwrite(img_out_path, night_vision_image)
            if not success: skipped_count += 1; continue
        except Exception as e: print(f"\nWarning: Error saving image '{img_out_path}': {e}. Skipping."); skipped_count += 1; continue

        try: shutil.copy2(label_in_path, label_out_path)
        except Exception as e:
            print(f"\nWarning: Failed to copy label file '{label_filename}' to output: {e}.")
            skipped_count += 1
            try:
                os.remove(img_out_path)
                print(f"Removed associated image '{img_out_path}' due to label copy failure.")
            except OSError as oe: print(f"Warning: Failed to remove image '{img_out_path}' after label copy failure: {oe}")
            continue # Skip incrementing processed_count if label copy failed

        processed_count += 1

    print()
    print("--- Processing Summary ---")
    print(f"Successfully processed: {processed_count} images and labels.")
    print(f"Skipped/Errors: {skipped_count} images (missing labels, read/proc/write errors, label copy failure).")
    print(f"Output saved in: {output_folder}")


# --- Main Execution ---

def main():
    # --- Configuration ---
    # <<<--- SET YOUR INPUT AND OUTPUT FOLDERS HERE --->>>
    INPUT_FOLDER = "./betterweapons/train"  # Use raw string (r"...") or double backslashes on Windows
    OUTPUT_FOLDER = "./train"

    # <<<--- ADJUST NIGHT VISION SIMULATION PARAMETERS HERE --->>>
    SIMULATION_SETTINGS = {
        # Overall brightness multiplier applied first (<1.0 darkens, 1.0=off)
        "exposure_factor": 0.1,

        # Std deviation for Gaussian noise (<=0 to disable)
        "noise_std_dev": 18.0,

        # Gamma correction (>1 brightens shadows, 1.0=off). Applied after exposure.
        "gamma": 2.0,

        # Gaussian blur kernel size (must be odd > 1, or <=1 to disable)
        "blur_ksize": 3,

        # Vignette effect strength [0.0-1.0] (<=0 to disable)
        "vignette_strength": 0.5,
        # Vignette radius scale (>1 pushes effect outwards)
        "vignette_radius_scale": 1.4,

        # Apply CLAHE contrast enhancement (applied after exposure, before gamma)
        "use_clahe": True,
        # CLAHE clip limit (only used if use_clahe is True)
        "clahe_clip": 2.0,
        # CLAHE tile grid size (only used if use_clahe is True)
        "clahe_grid": 8,

        # Tint color 'B,G,R' (e.g., "0,255,0" for green)
        "tint_color_str": "0,255,0",
        # Tint intensity [0.0-1.0] (<=0 to disable tinting)
        "tint_intensity": -1,
    }
    # --- End Configuration ---

    # --- Simple Validation (Optional but recommended) ---
    if SIMULATION_SETTINGS["blur_ksize"] > 1 and SIMULATION_SETTINGS["blur_ksize"] % 2 == 0:
        print(f"Warning: blur_ksize ({SIMULATION_SETTINGS['blur_ksize']}) is even. Blur requires an odd kernel size > 1. Disabling blur.")
        SIMULATION_SETTINGS["blur_ksize"] = 0 # Disable it
    elif SIMULATION_SETTINGS["blur_ksize"] == 1:
         SIMULATION_SETTINGS["blur_ksize"] = 0 # Standardize disabling to 0

    if not INPUT_FOLDER or not OUTPUT_FOLDER:
        print("Error: INPUT_FOLDER and OUTPUT_FOLDER must be set in the script.")
        return

    if not os.path.exists(INPUT_FOLDER):
         print(f"Error: Specified INPUT_FOLDER does not exist: {INPUT_FOLDER}")
         return

    # --- Run Processing ---
    process_dataset(INPUT_FOLDER, OUTPUT_FOLDER, SIMULATION_SETTINGS)

# --- Script Entry Point ---
if __name__ == "__main__":
    main()