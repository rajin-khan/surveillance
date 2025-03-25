from ultralytics import YOLO
import os
import shutil

model = YOLO("WeaponModel_v4.pt")

input_folder = "./betterweapons/test/images"
output_folder = "inference"
os.makedirs(output_folder, exist_ok=True)

image_files = [f for f in os.listdir(input_folder)
               if f.lower().endswith((".jpg", ".jpeg", ".png"))]

for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)

    results = model(image_path, save=True, save_txt=True)
    result = results[0]

    pred_dir = result.save_dir

    base_name = os.path.splitext(image_file)[0]
    pred_img = os.path.join(pred_dir, image_file)
    pred_txt = os.path.join(pred_dir, "labels", base_name + ".txt")

    if os.path.exists(pred_img):
        shutil.move(pred_img, os.path.join(output_folder, image_file))
    if os.path.exists(pred_txt):
        shutil.move(pred_txt, os.path.join(output_folder, base_name + ".txt"))