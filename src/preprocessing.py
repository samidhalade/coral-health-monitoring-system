import cv2
import numpy as np


def preprocess_image(image):
    """
    Prepare an uploaded coral image for visualisation
    and future model inference.

    Returns an RGB image.
    """

    image = np.array(image)

    # RGB → BGR
    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    # Resize
    image = cv2.resize(
        image,
        (640, 640)
    )

    # Contrast enhancement
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    enhanced = cv2.merge(
        (l, a, b)
    )

    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2RGB
    )

    return enhanced