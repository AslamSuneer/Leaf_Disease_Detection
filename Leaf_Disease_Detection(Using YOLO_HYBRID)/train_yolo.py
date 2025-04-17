from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolov8n.pt")

# Train
model.train(data="C:/Users/91730/YOLO_HYBRID/data.yaml", epochs=10, imgsz=128)