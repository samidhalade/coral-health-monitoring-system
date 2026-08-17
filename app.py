import streamlit as st
from PIL import Image
import os

from src.preprocessing import preprocess_image
from src.dataset_analysis import analyze_dataset


st.set_page_config(
    page_title="Coral Health Monitoring System",
    page_icon="🪸",
    layout="wide"
)


st.title("Coral Health Monitoring System")

st.write(
    "AI-based monitoring system for detecting and assessing "
    "coral health from underwater images."
)

st.divider()


# =========================================================
# PROJECT OVERVIEW
# =========================================================

st.header("Project Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Target Classes",
        "3"
    )

with col2:
    st.metric(
        "Current Module",
        "Image Analysis"
    )

with col3:
    st.metric(
        "Future Module",
        "Water Quality"
    )


st.divider()


# =========================================================
# DATASET
# =========================================================

st.header("Dataset")

try:

    df = analyze_dataset()

    st.dataframe(
        df,
        use_container_width=True
    )

    st.bar_chart(
        df.set_index("Dataset Split")["Images"]
    )

except Exception as e:

    st.warning(
        "Dataset analysis could not be loaded."
    )


st.divider()


# =========================================================
# IMAGE ANALYSIS
# =========================================================

st.header("Coral Image Analysis")

uploaded_file = st.file_uploader(
    "Upload an underwater coral image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.subheader("Preprocessed Image")

        processed = preprocess_image(image)

        # Convert BGR → RGB
        processed_rgb = processed[:, :, ::-1]

        st.image(
            processed_rgb,
            use_container_width=True
        )

    st.divider()

    st.subheader("Analysis Status")

    st.info(
        "Image preprocessing completed successfully."
    )

    st.write(
        "**Preprocessing performed:**"
    )

    st.write(
        "- Image resizing to 640 × 640 pixels\n"
        "- Color-space conversion\n"
        "- Contrast enhancement using CLAHE\n"
        "- Image preparation for YOLO-based detection"
    )

    st.divider()

    st.subheader("Classification Module")

    model_path = (
        "models/coral_health_yolov8/"
        "weights/best.pt"
    )

    if os.path.exists(model_path):

        st.success(
            "Trained YOLO model detected. "
            "Prediction module is ready."
        )

    else:

        st.warning(
            "YOLO model training is currently in progress. "
            "The trained model will be integrated here after "
            "training is completed."
        )


st.divider()


# =========================================================
# TARGET CLASSES
# =========================================================

st.header("Coral Health Categories")

c1, c2, c3 = st.columns(3)

with c1:

    st.subheader("Healthy Coral")

    st.write(
        "Coral showing visual characteristics "
        "associated with healthy reef conditions."
    )

with c2:

    st.subheader("Bleached Coral")

    st.write(
        "Coral showing visual signs of bleaching "
        "and environmental stress."
    )

with c3:

    st.subheader("Dead Coral")

    st.write(
        "Coral showing severe degradation or "
        "loss of living coral tissue."
    )


st.divider()


# =========================================================
# FUTURE MODULE
# =========================================================

st.header("Future Environmental Analysis")

st.write(
    "The next stage of the project will integrate "
    "environmental parameters to investigate possible "
    "causes of coral health deterioration."
)

st.write(
    "**Planned parameters:**"
)

st.write(
    "- Temperature\n"
    "- pH\n"
    "- Salinity\n"
    "- Dissolved Oxygen\n"
    "- Nitrate\n"
    "- Phosphate"
)


st.info(
    "Final system architecture: "
    "Image Detection → Health Classification → "
    "Environmental Analysis → Recommendations"
)