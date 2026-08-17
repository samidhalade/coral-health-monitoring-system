import os
import tempfile

import streamlit as st
from PIL import Image

from src.preprocessing import preprocess_image
from src.dataset_analysis import analyze_dataset
from src.prediction import load_model, predict_coral
from src.health_report import generate_health_report


st.set_page_config(
    page_title="Coral Health Monitoring System",
    page_icon="🪸",
    layout="wide"
)

MODEL_PATH = "models/coral_health_yolov8/weights/best.pt"

st.title("Coral Health Monitoring System")
st.write(
    "AI-based coral detection and visual health assessment system."
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Dataset Analysis",
        "Coral Health Analysis",
        "Environmental Analysis"
    ]
)


if page == "Dashboard":

    st.header("Project Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Coral Categories", "3")

    with col2:
        st.metric("Detection Model", "YOLOv8")

    with col3:
        st.metric("Analysis Type", "Image Based")

    st.divider()

    st.subheader("Coral Categories")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Healthy Coral")
        st.write(
            "Coral showing visual characteristics "
            "associated with healthy conditions."
        )

    with col2:
        st.markdown("### Bleached Coral")
        st.write(
            "Coral showing visual characteristics "
            "associated with bleaching and stress."
        )

    with col3:
        st.markdown("### Dead Coral")
        st.write(
            "Coral showing severe degradation or "
            "loss of living coral tissue."
        )

    st.divider()

    st.subheader("System Workflow")

    st.write(
        "Upload Image → Coral Detection → Health Classification "
        "→ Health Assessment → Recommendations → Environmental Analysis"
    )


elif page == "Dataset Analysis":

    st.header("Dataset Analysis")

    try:

        df = analyze_dataset()

        if df.empty:

            st.warning(
                "No dataset records were found."
            )

        else:

            st.dataframe(
                df,
                use_container_width=True
            )

            if "Dataset Split" in df.columns and "Images" in df.columns:

                st.subheader("Dataset Distribution")

                st.bar_chart(
                    df.set_index("Dataset Split")["Images"]
                )

    except Exception as error:

        st.error(
            f"Dataset analysis failed: {error}"
        )


elif page == "Coral Health Analysis":

    st.header("Coral Health Assessment")

    st.write(
        "Upload a coral image to analyse its visual health condition."
    )

    uploaded_file = st.file_uploader(
        "Upload Coral Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.subheader("Uploaded Coral Image")

        st.image(
            image,
            use_container_width=True
        )

        if not os.path.exists(MODEL_PATH):

            st.warning(
                "The trained YOLO model is not available yet."
            )

            st.info(
                "Add a valid best.pt file to "
                "models/coral_health_yolov8/weights/ "
                "to enable real coral classification."
            )

            st.stop()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as temp_file:

            image.save(
                temp_file.name
            )

            image_path = temp_file.name

        with st.spinner("Analysing coral image..."):

            model = load_model(
                MODEL_PATH
            )

        if model is None:

            st.warning(
                "The YOLO model could not be loaded."
            )

            st.info(
                "The best.pt file may be incomplete or corrupted. "
                "A valid trained model is required for classification."
            )

            st.stop()

        with st.spinner("Detecting coral health condition..."):

            prediction = predict_coral(
                model,
                image_path
            )

        if prediction is None:

            st.warning(
                "No coral was detected in this image."
            )

            st.info(
                "Try uploading a clear underwater coral image."
            )

            st.stop()

        report = generate_health_report(
            prediction["class_name"],
            prediction["confidence"]
        )

        st.divider()

        st.header("Coral Health Assessment Report")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Detected Category",
                report["category"]
            )

        with col2:

            st.metric(
                "Health Status",
                report["health_status"]
            )

        with col3:

            st.metric(
                "Confidence",
                f'{report["confidence"]}%'
            )

        st.divider()

        st.subheader("Risk Level")

        risk = report["risk_level"]

        if risk == "Low":

            st.success(risk)

        elif risk == "High":

            st.warning(risk)

        else:

            st.error(risk)

        st.subheader("Visual Assessment")

        st.write(
            report["description"]
        )

        st.subheader("Health Interpretation")

        st.write(
            report["implications"]
        )

        st.subheader("Recommended Actions")

        for recommendation in report["recommendations"]:

            st.write(
                f"• {recommendation}"
            )

        st.divider()

        st.subheader("Environmental Cause Analysis")

        st.info(
            report["environmental_analysis"]
        )

        st.write(
            "The following environmental parameters "
            "will be used in the future analysis module:"
        )

        for parameter in report["required_environmental_data"]:

            st.write(
                f"• {parameter}"
            )

        st.divider()

        st.caption(
            f'Analysis performed: {report["analysis_time"]}'
        )


elif page == "Environmental Analysis":

    st.header("Environmental Cause Analysis")

    st.info(
        "This module will be integrated after the "
        "environmental dataset is added."
    )

    st.subheader("Planned Environmental Parameters")

    parameters = [
        "Water Temperature",
        "pH",
        "Salinity",
        "Dissolved Oxygen",
        "Nitrate",
        "Phosphate"
    ]

    for parameter in parameters:

        st.write(
            f"• {parameter}"
        )

    st.divider()

    st.subheader("Future Analysis")

    st.write(
        "Environmental parameters will be analysed "
        "alongside coral health classifications to "
        "identify factors associated with coral stress "
        "and deterioration."
    )