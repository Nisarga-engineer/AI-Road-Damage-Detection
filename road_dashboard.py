import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import urllib.parse
import tempfile
import os


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Road Damage Detection",
    page_icon="🚧",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 17px;
    opacity: 0.75;
    margin-bottom: 25px;
}

.section-title {
    font-size: 25px;
    font-weight: 650;
    margin-top: 25px;
    margin-bottom: 15px;
}

.metric-card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.25);
    text-align: center;
}

.metric-number {
    font-size: 30px;
    font-weight: 700;
}

.metric-label {
    font-size: 14px;
    opacity: 0.7;
}

.pothole-card {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 12px;
}

.footer {
    text-align: center;
    opacity: 0.6;
    margin-top: 40px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🚧 AI Road Damage Detection & Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered pothole detection, severity assessment and road condition analysis'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = (
    r"C:\Users\darsh\OneDrive\Desktop\AI Road Damage System"
    r"\runs\segment\train-6\weights\best.pt"
)

model = YOLO(MODEL_PATH)


# =========================================================
# INPUT TYPE
# =========================================================

st.markdown(
    '<div class="section-title">📤 Upload Road Data</div>',
    unsafe_allow_html=True
)

input_type = st.radio(
    "Choose input type",
    ["🖼️ Road Image", "🎥 Road Video"],
    horizontal=True
)


# =========================================================
# IMAGE ANALYSIS FUNCTION
# =========================================================

def calculate_severity(area_percentage, confidence):

    # Size score
    if area_percentage < 1:
        size_score = 10

    elif area_percentage < 3:
        size_score = 20

    elif area_percentage < 5:
        size_score = 30

    elif area_percentage < 10:
        size_score = 40

    else:
        size_score = 50

    # Confidence score
    confidence_score = confidence * 20

    # Final severity
    severity_score = int(
        min(
            ((size_score + confidence_score) / 70) * 100,
            100
        )
    )

    # Severity category
    if severity_score < 30:
        severity = "LOW"
        priority = "LOW"

    elif severity_score < 60:
        severity = "MEDIUM"
        priority = "MEDIUM"

    else:
        severity = "HIGH"
        priority = "HIGH"

    return severity_score, severity, priority


# =========================================================
# IMAGE MODE
# =========================================================

if input_type == "🖼️ Road Image":

    uploaded = st.file_uploader(
        "Choose a road image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear road image."
    )

    if uploaded is not None:

        st.success("Image uploaded successfully!")

        image = Image.open(uploaded).convert("RGB")

        image_array = np.array(image)

        st.markdown(
            '<div class="section-title">🖼️ Image Analysis</div>',
            unsafe_allow_html=True
        )

        with st.spinner("Running AI pothole detection..."):

            results = model.predict(
                source=image_array,
                conf=0.15,
                device=0
            )

        result = results[0]

        count = len(result.boxes)

        output = result.plot()

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Original Road Image",
                use_container_width=True
            )

        with col2:

            st.image(
                output,
                caption="AI Pothole Detection",
                use_container_width=True
            )

        if count == 0:

            st.success("✅ No potholes detected.")

        else:

            st.success(
                f"🤖 AI detected {count} pothole(s)"
            )

            high_count = 0
            medium_count = 0
            low_count = 0

            severity_scores = []

            pothole_results = []

            if result.masks is not None:

                image_height, image_width = image_array.shape[:2]

                total_image_area = (
                    image_height * image_width
                )

                for i, mask_data in enumerate(
                    result.masks.data
                ):

                    mask = mask_data.cpu().numpy()

                    mask = cv2.resize(
                        mask,
                        (image_width, image_height),
                        interpolation=cv2.INTER_NEAREST
                    )

                    mask = mask > 0.5

                    pothole_area = int(mask.sum())

                    area_percentage = (
                        pothole_area /
                        total_image_area
                    ) * 100

                    confidence = float(
                        result.boxes.conf[i]
                    )

                    severity_score, severity, priority = (
                        calculate_severity(
                            area_percentage,
                            confidence
                        )
                    )

                    severity_scores.append(
                        severity_score
                    )

                    if severity == "HIGH":
                        high_count += 1

                    elif severity == "MEDIUM":
                        medium_count += 1

                    else:
                        low_count += 1

                    pothole_results.append({
                        "number": i + 1,
                        "area": pothole_area,
                        "percentage": area_percentage,
                        "confidence": confidence * 100,
                        "score": severity_score,
                        "severity": severity,
                        "priority": priority
                    })


            # =================================================
            # IMAGE SUMMARY
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '📊 Detection Summary'
                '</div>',
                unsafe_allow_html=True
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "Total Potholes",
                    count
                )

            with c2:
                st.metric(
                    "High Severity",
                    high_count
                )

            with c3:
                st.metric(
                    "Medium Severity",
                    medium_count
                )

            with c4:
                st.metric(
                    "Low Severity",
                    low_count
                )


            # =================================================
            # INDIVIDUAL ANALYSIS
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '🕳️ Individual Pothole Analysis'
                '</div>',
                unsafe_allow_html=True
            )

            for p in pothole_results:

                st.markdown(
                    f"""
                    <div class="pothole-card">

                    <h3>🕳️ Pothole {p["number"]}</h3>

                    <b>Area:</b> {p["area"]:,} pixels<br>

                    <b>Area Percentage:</b>
                    {p["percentage"]:.2f}%<br>

                    <b>AI Confidence:</b>
                    {p["confidence"]:.1f}%<br>

                    <b>Severity Score:</b>
                    {p["score"]}/100<br>

                    <b>Severity:</b>
                    {p["severity"]}<br>

                    <b>Repair Priority:</b>
                    {p["priority"]}

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # =================================================
            # OVERALL ROAD ASSESSMENT
            # =================================================

            average_severity = (
                sum(severity_scores) /
                len(severity_scores)
            )

            if (
                high_count >= 2
                or average_severity >= 70
            ):

                condition = "POOR"
                priority = "HIGH"

                recommendation = (
                    "Immediate road maintenance recommended."
                )

            elif (
                high_count == 1
                or average_severity >= 40
            ):

                condition = "MODERATE"
                priority = "MEDIUM"

                recommendation = (
                    "Road inspection and maintenance recommended."
                )

            else:

                condition = "GOOD"
                priority = "LOW"

                recommendation = (
                    "Routine monitoring recommended."
                )


            st.divider()

            st.markdown(
                '<div class="section-title">'
                '🚧 Overall Road Assessment'
                '</div>',
                unsafe_allow_html=True
            )

            a1, a2, a3 = st.columns(3)

            with a1:

                st.metric(
                    "Average Severity",
                    f"{average_severity:.1f}/100"
                )

            with a2:

                st.metric(
                    "Road Condition",
                    condition
                )

            with a3:

                st.metric(
                    "Repair Priority",
                    priority
                )

            st.info(
                f"💡 **Recommended Action:** "
                f"{recommendation}"
            )


            # =================================================
            # WHATSAPP
            # =================================================

            whatsapp_message = (
                "🚧 AI Road Damage Analysis\n\n"
                f"Total Potholes: {count}\n"
                f"High Severity: {high_count}\n"
                f"Medium Severity: {medium_count}\n"
                f"Low Severity: {low_count}\n"
                f"Average Severity: "
                f"{average_severity:.1f}/100\n"
                f"Road Condition: {condition}\n"
                f"Repair Priority: {priority}\n\n"
                f"Recommendation: {recommendation}"
            )

            encoded_message = urllib.parse.quote(
                whatsapp_message
            )

            whatsapp_url = (
                "https://wa.me/?text="
                + encoded_message
            )

            st.markdown(
                '<div class="section-title">'
                '📱 Share Result'
                '</div>',
                unsafe_allow_html=True
            )

            st.link_button(
                "📱 Share Analysis on WhatsApp",
                whatsapp_url
            )


# =========================================================
# VIDEO MODE
# =========================================================

else:

    uploaded_video = st.file_uploader(
        "Choose a road video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a road video for pothole detection."
    )

    if uploaded_video is not None:

        st.success("Video uploaded successfully!")

        # -------------------------------------------------
        # SAVE TEMP VIDEO
        # -------------------------------------------------

        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_file.write(
            uploaded_video.read()
        )

        input_path = input_file.name

        input_file.close()


        # -------------------------------------------------
        # VIDEO PREVIEW
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            '🎥 Video Analysis'
            '</div>',
            unsafe_allow_html=True
        )

        st.video(
            input_path
        )


        # -------------------------------------------------
        # OPEN VIDEO
        # -------------------------------------------------

        cap = cv2.VideoCapture(
            input_path
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )


        if fps == 0:
            fps = 25


        # -------------------------------------------------
        # OUTPUT VIDEO
        # -------------------------------------------------

        output_path = os.path.join(
            tempfile.gettempdir(),
            "ai_detected_road_video.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )


        # -------------------------------------------------
        # VIDEO VARIABLES
        # -------------------------------------------------

        processed_frames = 0

        total_detections = 0

        max_potholes = 0

        high_count = 0

        medium_count = 0

        low_count = 0

        all_severity_scores = []

        progress = st.progress(0)

        status_text = st.empty()


        # -------------------------------------------------
        # PROCESS VIDEO
        # -------------------------------------------------

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            processed_frames += 1


            # ---------------------------------------------
            # YOLO
            # ---------------------------------------------

            results = model.predict(
                source=frame,
                conf=0.15,
                device=0,
                verbose=False
            )

            result = results[0]

            frame_count = len(
                result.boxes
            )

            total_detections += frame_count

            max_potholes = max(
                max_potholes,
                frame_count
            )


            # ---------------------------------------------
            # FRAME SEVERITY
            # ---------------------------------------------

            if (
                result.masks is not None
                and frame_count > 0
            ):

                for i, mask_data in enumerate(
                    result.masks.data
                ):

                    mask = (
                        mask_data
                        .cpu()
                        .numpy()
                    )

                    mask = cv2.resize(
                        mask,
                        (width, height),
                        interpolation=cv2.INTER_NEAREST
                    )

                    mask = mask > 0.5

                    pothole_area = mask.sum()

                    frame_area = (
                        width * height
                    )

                    area_percentage = (
                        pothole_area /
                        frame_area
                    ) * 100

                    confidence = float(
                        result.boxes.conf[i]
                    )

                    severity_score, severity, priority = (
                        calculate_severity(
                            area_percentage,
                            confidence
                        )
                    )

                    all_severity_scores.append(
                        severity_score
                    )

                    if severity == "HIGH":
                        high_count += 1

                    elif severity == "MEDIUM":
                        medium_count += 1

                    else:
                        low_count += 1


            # ---------------------------------------------
            # DRAW DETECTIONS
            # ---------------------------------------------

            annotated_frame = result.plot()

            out.write(
                annotated_frame
            )


            # ---------------------------------------------
            # PROGRESS
            # ---------------------------------------------

            progress_value = (
                processed_frames /
                total_frames
            )

            progress.progress(
                min(progress_value, 1.0)
            )

            status_text.text(
                f"Processing frame "
                f"{processed_frames}/{total_frames} | "
                f"Potholes in current frame: "
                f"{frame_count}"
            )


        # -------------------------------------------------
        # RELEASE
        # -------------------------------------------------

        cap.release()

        out.release()

        progress.empty()

        status_text.empty()


        st.success(
            "🎥 Video analysis completed!"
        )


        # =================================================
        # VIDEO SUMMARY
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '📊 Video Detection Summary'
            '</div>',
            unsafe_allow_html=True
        )


        if all_severity_scores:

            average_video_severity = (
                sum(all_severity_scores) /
                len(all_severity_scores)
            )

        else:

            average_video_severity = 0


        # -------------------------------------------------
        # OVERALL VIDEO CONDITION
        # -------------------------------------------------

        if (
            high_count >= 5
            or average_video_severity >= 70
        ):

            video_condition = "POOR"
            video_priority = "HIGH"

            video_recommendation = (
                "Immediate road inspection and maintenance recommended."
            )

        elif (
            high_count > 0
            or average_video_severity >= 40
        ):

            video_condition = "MODERATE"
            video_priority = "MEDIUM"

            video_recommendation = (
                "Road inspection and maintenance recommended."
            )

        elif total_detections > 0:

            video_condition = "GOOD"
            video_priority = "LOW"

            video_recommendation = (
                "Routine road monitoring recommended."
            )

        else:

            video_condition = "GOOD"
            video_priority = "LOW"

            video_recommendation = (
                "No significant pothole damage detected."
            )


        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------

        v1, v2, v3, v4 = st.columns(4)

        with v1:

            st.metric(
                "Total Frames",
                processed_frames
            )

        with v2:

            st.metric(
                "Total Detections",
                total_detections
            )

        with v3:

            st.metric(
                "Maximum Potholes / Frame",
                max_potholes
            )

        with v4:

            st.metric(
                "Average Severity",
                f"{average_video_severity:.1f}/100"
            )


        # =================================================
        # SEVERITY SUMMARY
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '🚨 Video Severity Analysis'
            '</div>',
            unsafe_allow_html=True
        )


        s1, s2, s3 = st.columns(3)

        with s1:

            st.metric(
                "🔴 High Severity",
                high_count
            )

        with s2:

            st.metric(
                "🟠 Medium Severity",
                medium_count
            )

        with s3:

            st.metric(
                "🟢 Low Severity",
                low_count
            )


        # =================================================
        # ROAD ASSESSMENT
        # =================================================

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '🚧 Overall Road Assessment'
            '</div>',
            unsafe_allow_html=True
        )


        r1, r2, r3 = st.columns(3)

        with r1:

            st.metric(
                "Average Severity",
                f"{average_video_severity:.1f}/100"
            )

        with r2:

            st.metric(
                "Road Condition",
                video_condition
            )

        with r3:

            st.metric(
                "Repair Priority",
                video_priority
            )


        st.info(
            f"💡 **Recommended Action:** "
            f"{video_recommendation}"
        )


        # =================================================
        # OUTPUT VIDEO
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '🎥 AI Detection Result'
            '</div>',
            unsafe_allow_html=True
        )


        if os.path.exists(output_path):

            st.video(
                output_path
            )


            # ------------------------------------------------
            # DOWNLOAD BUTTON
            # ------------------------------------------------

            with open(
                output_path,
                "rb"
            ) as video_file:

                st.download_button(
                    label="⬇️ Download Detected Video",
                    data=video_file,
                    file_name="AI_detected_road_video.mp4",
                    mime="video/mp4"
                )


        # =================================================
        # WHATSAPP
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '📱 Share Video Analysis'
            '</div>',
            unsafe_allow_html=True
        )


        whatsapp_message = (
            "🚧 AI Road Damage Video Analysis\n\n"
            f"Total Frames: {processed_frames}\n"
            f"Total Pothole Detections: "
            f"{total_detections}\n"
            f"Maximum Potholes / Frame: "
            f"{max_potholes}\n"
            f"High Severity: {high_count}\n"
            f"Medium Severity: {medium_count}\n"
            f"Low Severity: {low_count}\n"
            f"Average Severity: "
            f"{average_video_severity:.1f}/100\n"
            f"Road Condition: {video_condition}\n"
            f"Repair Priority: {video_priority}\n\n"
            f"Recommendation: "
            f"{video_recommendation}"
        )


        encoded_message = urllib.parse.quote(
            whatsapp_message
        )


        whatsapp_url = (
            "https://wa.me/?text="
            + encoded_message
        )


        st.link_button(
            "📱 Share Video Analysis on WhatsApp",
            whatsapp_url
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
    🚧 AI Road Damage Detection & Analysis System<br>
    Powered by YOLO Segmentation & Computer Vision
    </div>
    """,
    unsafe_allow_html=True
)