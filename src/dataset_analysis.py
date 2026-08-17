import os
import pandas as pd


DATASET_PATH = "dataset/coral_dataset"


def analyze_dataset():

    records = []

    for split in ["train", "valid", "test"]:

        image_folder = os.path.join(
            DATASET_PATH,
            split,
            "images"
        )

        label_folder = os.path.join(
            DATASET_PATH,
            split,
            "labels"
        )

        if not os.path.exists(image_folder):
            continue

        images = [
            f for f in os.listdir(image_folder)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]

        labels = []

        if os.path.exists(label_folder):
            labels = [
                f for f in os.listdir(label_folder)
                if f.endswith(".txt")
            ]

        records.append({
            "Dataset Split": split,
            "Images": len(images),
            "Labels": len(labels)
        })

    return pd.DataFrame(records)


if __name__ == "__main__":

    df = analyze_dataset()

    print("\nCORAL DATASET ANALYSIS")
    print("=" * 40)

    print(df.to_string(index=False))