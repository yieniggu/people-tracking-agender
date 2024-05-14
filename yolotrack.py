from ultralytics import YOLO

# Load an official or custom model
model = YOLO('yolov8n.pt')  # Load an official Detect model

results = model.track("test.avi", show=True, persist=True, tracker="bytetrack.yaml")