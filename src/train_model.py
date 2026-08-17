from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="dataset/coral_dataset/data.yaml",
    epochs=3,
    imgsz=416,
    batch=4,
    workers=2,
    project="models",
    name="coral_health_yolov8_demo"
)

print("Training completed.")