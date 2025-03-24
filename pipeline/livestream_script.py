import cv2
from ultralytics import YOLO
import os
from dotenv import load_dotenv
import sys

load_dotenv()

rtsp_url = os.getenv("CAM1")

if not rtsp_url:
    print("[ERROR] RTSP_URL not set in environment.")
    sys.exit(1)

model_path = "path/to/model.pt"

frame_size = (640, 480)

try:
    model = YOLO(model_path)
    print(f"[INFO] Loaded model from {model_path}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    sys.exit(1)

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print(f"[ERROR] Unable to open RTSP stream: {rtsp_url}")
    sys.exit(1)

print(f"[INFO] Streaming from: {rtsp_url}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Frame grab failed.")
        continue

    if frame_size:
        frame = cv2.resize(frame, frame_size)

    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("YOLOv8 Inference (RTSP)", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()