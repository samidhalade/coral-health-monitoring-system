from ultralytics import YOLO
import sys
import os

MODEL_PATH = "models/coral_health_yolov8/weights/best.pt"

model = YOLO(MODEL_PATH)

if len(sys.argv) < 2:

    print("Please provide an image path.")
    print("Example:")
    print("python src/predict.py test_coral.jpg")
    sys.exit()

image_path = sys.argv[1]

results = model.predict(
    source=image_path,
    conf=0.25,
    save=True
)

for result in results:

    print("\n" + "=" * 50)
    print("CORAL HEALTH DETECTION")
    print("=" * 50)

    if result.boxes is None or len(result.boxes) == 0:

        print("No coral detected.")

    else:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            print(f"Coral Condition : {class_name}")
            print(f"Confidence      : {confidence * 100:.2f}%")
            print("-" * 50)