from datetime import datetime

from src.prediction import CLASS_INFORMATION


def generate_health_report(
    class_name,
    confidence
):
    """
    Generate a structured coral health report.
    """

    information = CLASS_INFORMATION.get(class_name)

    if information is None:
        return None

    report = {
        "analysis_time": datetime.now().strftime(
            "%d %B %Y, %I:%M %p"
        ),

        "category": class_name,

        "confidence": round(
            confidence * 100,
            2
        ),

        "health_status": information["health_status"],

        "risk_level": information["risk_level"],

        "description": information["description"],

        "implications": information["implications"],

        "recommendations": information["recommendations"],

        "environmental_analysis": (
            "Environmental cause analysis is not available "
            "from image data alone."
        ),

        "required_environmental_data": [
            "Water temperature",
            "pH",
            "Salinity",
            "Dissolved oxygen",
            "Nitrate",
            "Phosphate"
        ]
    }

    return report