from ultralytics import YOLO
from pathlib import Path


CLASS_INFORMATION = {
    "Healthy Coral": {
        "health_status": "Good",
        "risk_level": "Low",
        "description": (
            "The coral has been classified as healthy based on "
            "the visual characteristics detected in the image."
        ),
        "implications": (
            "The coral currently shows visual characteristics "
            "associated with a healthy condition."
        ),
        "recommendations": [
            "Continue routine monitoring.",
            "Record changes in coral appearance over time.",
            "Monitor surrounding environmental conditions.",
            "Maintain regular reef health assessments."
        ]
    },

    "Bleached Coral": {
        "health_status": "Stressed",
        "risk_level": "High",
        "description": (
            "The coral shows visual characteristics associated "
            "with coral bleaching."
        ),
        "implications": (
            "Bleaching is a stress response and may indicate "
            "that the coral is experiencing unfavorable conditions. "
            "Bleached coral can recover if the stress is reduced, "
            "but prolonged stress can result in mortality."
        ),
        "recommendations": [
            "Increase monitoring frequency.",
            "Check surrounding water temperature.",
            "Assess local water-quality conditions.",
            "Monitor the coral for further tissue loss or deterioration.",
            "Investigate possible environmental stressors."
        ]
    },

    "Dead Coral": {
        "health_status": "Critical / Degraded",
        "risk_level": "Very High",
        "description": (
            "The coral has been classified as dead based on "
            "the visual characteristics detected in the image."
        ),
        "implications": (
            "The image indicates severe coral degradation. "
            "The exact cause of mortality cannot be determined "
            "from the image alone."
        ),
        "recommendations": [
            "Document the affected coral and surrounding reef area.",
            "Monitor nearby coral colonies.",
            "Assess local water-quality conditions.",
            "Investigate possible biological and environmental stressors.",
            "Record the observation for long-term reef monitoring."
        ]
    }
}


def load_model(model_path):
    """
    Safely load the trained YOLO model.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        return None

    # Check whether the file is empty
    if model_path.stat().st_size == 0:
        return None

    try:
        return YOLO(str(model_path))

    except Exception:
        return None

def predict_coral(model, image_path):
    """
    Run YOLO prediction on a coral image.

    Returns the class name, confidence and bounding boxes.
    """

    if model is None:
        return None

    results = model.predict(
        source=image_path,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    if not results:
        return None

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return None

    confidences = result.boxes.conf.cpu().numpy()
    best_index = confidences.argmax()

    confidence = float(confidences[best_index])

    class_id = int(
        result.boxes.cls.cpu().numpy()[best_index]
    )

    class_name = model.names[class_id]

    return {
        "class_name": class_name,
        "confidence": confidence,
        "result": result
    }