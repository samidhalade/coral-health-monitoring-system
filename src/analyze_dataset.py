import os
from collections import Counter

DATASET = "dataset/coral_dataset"

splits = {
    "train": "train",
    "validation": "valid",
    "test": "test"
}

for name, folder in splits.items():

    image_dir = os.path.join(DATASET, folder, "images")
    label_dir = os.path.join(DATASET, folder, "labels")

    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    label_files = [
        f for f in os.listdir(label_dir)
        if f.lower().endswith(".txt")
    ]

    print("\n" + "=" * 50)
    print(name.upper())
    print("=" * 50)

    print("Images:", len(image_files))
    print("Labels:", len(label_files))

    class_counter = Counter()

    for label_file in label_files:

        path = os.path.join(label_dir, label_file)

        with open(path, "r") as file:

            for line in file:

                parts = line.strip().split()

                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_counter[class_id] += 1

    print("Class distribution:")

    for class_id, count in sorted(class_counter.items()):

        class_names = {
            0: "Bleached Coral",
            1: "Dead Coral",
            2: "Healthy Coral"
        }

        print(
            class_id,
            class_names.get(class_id, "Unknown"),
            ":", count
        )