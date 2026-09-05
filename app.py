import os
import cv2
import torch
import numpy as np
import streamlit as st

from src.preprocessing.video_processor import VideoProcessor
from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.face_processor import FaceProcessor
from src.preprocessing.face_aligner import FaceAligner

from src.models.efficientnet_feature_extractor import (
    EfficientNetFeatureExtractor
)

from src.models.temporal_transformer import TemporalTransformer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DeepGuard",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH = (
    "results/models/frame_sampling_transformer.pt"
)

NUM_FRAMES = 64
IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ DeepGuard")

st.subheader(
    "Deepfake Detection & Media Forensics"
)

st.write(
    "Analyze video authenticity using spatial "
    "and temporal deep-learning features."
)

st.divider()


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    # --------------------------------------------------------
    # EfficientNet-B0
    #
    # IMPORTANT:
    # Your EfficientNetFeatureExtractor __init__()
    # takes NO arguments.
    # --------------------------------------------------------

    feature_extractor = (
        EfficientNetFeatureExtractor()
    )

    feature_extractor = (
        feature_extractor.to(DEVICE)
    )

    feature_extractor.eval()

    # --------------------------------------------------------
    # Temporal Transformer
    # --------------------------------------------------------

    temporal_model = TemporalTransformer(
        input_size=1280,
        hidden_size=256,
        num_heads=8,
        num_layers=2,
        dropout=0.3,
        max_frames=64,
    )

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model checkpoint not found:\n"
            f"{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    temporal_model.load_state_dict(
        checkpoint
    )

    temporal_model = (
        temporal_model.to(DEVICE)
    )

    temporal_model.eval()

    return (
        feature_extractor,
        temporal_model
    )


# ============================================================
# VIDEO METADATA
# ============================================================

def get_video_metadata(video_path):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open uploaded video."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
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

    cap.release()

    if fps > 0:

        duration = (
            frame_count / fps
        )

    else:

        duration = 0.0

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
    }


# ============================================================
# PREPARE VIDEO
# ============================================================

def prepare_video(video_path):

    """
    Prepare video for inference.

    Video
      ↓
    64 sampled frames
      ↓
    Face detection
      ↓
    Largest face
      ↓
    Face processing
      ↓
    Face alignment
      ↓
    RGB
      ↓
    [0,1] normalization
      ↓
    Tensor [64,3,224,224]
    """

    video_processor = (
        VideoProcessor(video_path)
    )

    face_detector = FaceDetector()

    # IMPORTANT:
    # Match the training configuration.
    face_processor = FaceProcessor(
        output_size=(224, 224),
        margin=0.2,
    )

    face_aligner = FaceAligner()

    processed_faces = []

    detection_count = 0

    try:

        # ----------------------------------------------------
        # Sample 64 frames
        # ----------------------------------------------------

        frames = (
            video_processor.sample_frames(
                num_frames=NUM_FRAMES
            )
        )

        # ----------------------------------------------------
        # Process each frame
        # ----------------------------------------------------

        for frame in frames:

            # Face detection
            faces = (
                face_detector.detect(frame)
            )

            if not faces:
                continue

            detection_count += 1

            # ------------------------------------------------
            # Largest face
            # ------------------------------------------------

            largest_face = max(
                faces,
                key=lambda box:
                    box[2] * box[3]
            )

            # ------------------------------------------------
            # Face processing
            # ------------------------------------------------

            processed_face_list = (
                face_processor.process_frame(
                    frame,
                    [largest_face]
                )
            )

            if not processed_face_list:
                continue

            face = processed_face_list[0]

            # ------------------------------------------------
            # Face alignment
            # ------------------------------------------------

            aligned_face = (
                face_aligner.align(face)
            )

            if aligned_face is None:
                continue

            # ------------------------------------------------
            # BGR -> RGB
            # ------------------------------------------------

            aligned_face = cv2.cvtColor(
                aligned_face,
                cv2.COLOR_BGR2RGB
            )

            processed_faces.append(
                aligned_face
            )

    finally:

        face_aligner.close()

        video_processor.close()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not processed_faces:

        raise RuntimeError(
            "No valid faces could be detected "
            "in the video."
        )

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    faces_array = np.stack(
        processed_faces
    )

    # --------------------------------------------------------
    # Convert:
    #
    # [N,H,W,C]
    #
    # to:
    #
    # [N,C,H,W]
    #
    # IMPORTANT:
    # Match training normalization.
    # --------------------------------------------------------

    faces_tensor = (
        torch.from_numpy(
            faces_array
        )
        .permute(
            0,
            3,
            1,
            2
        )
        .float()
        / 255.0
    )

    # --------------------------------------------------------
    # Pad to 64 frames
    # --------------------------------------------------------

    if faces_tensor.shape[0] < NUM_FRAMES:

        last_face = (
            faces_tensor[-1:]
            .clone()
        )

        padding_count = (
            NUM_FRAMES
            - faces_tensor.shape[0]
        )

        padding = last_face.repeat(
            padding_count,
            1,
            1,
            1
        )

        faces_tensor = torch.cat(
            [
                faces_tensor,
                padding
            ],
            dim=0
        )

    # --------------------------------------------------------
    # Safety trim
    # --------------------------------------------------------

    faces_tensor = (
        faces_tensor[:NUM_FRAMES]
    )

    return (
        faces_tensor,
        detection_count
    )


# ============================================================
# ANALYZE VIDEO
# ============================================================

def analyze_video(video_path):

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    (
        feature_extractor,
        temporal_model
    ) = load_models()

    # --------------------------------------------------------
    # Prepare faces
    # --------------------------------------------------------

    (
        faces_tensor,
        detection_count
    ) = prepare_video(
        video_path
    )

    faces_tensor = (
        faces_tensor.to(DEVICE)
    )

    # --------------------------------------------------------
    # Extract EfficientNet features
    # --------------------------------------------------------

    spatial_features = []

    with torch.no_grad():

        for face in faces_tensor:

            # Face is already [0,1]
            processed_face = (
                feature_extractor.preprocess(
                    face
                )
                .unsqueeze(0)
                .to(DEVICE)
            )

            feature = (
                feature_extractor(
                    processed_face
                )
            )

            spatial_features.append(
                feature
            )

    # --------------------------------------------------------
    # Combine features
    #
    # [64,1280]
    #      ↓
    # [1,64,1280]
    # --------------------------------------------------------

    spatial_features = (
        torch.cat(
            spatial_features,
            dim=0
        )
        .unsqueeze(0)
    )

    # --------------------------------------------------------
    # Temporal Transformer
    # --------------------------------------------------------

    with torch.no_grad():

        logits = temporal_model(
            spatial_features
        )

        fake_probability = (
            torch.sigmoid(
                logits
            )
            .item()
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if fake_probability >= 0.5:

        prediction = "FAKE"

    else:

        prediction = "REAL"

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    authenticity_score = (
        1.0 - fake_probability
    ) * 100.0

    fake_probability_percent = (
        fake_probability * 100.0
    )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    if fake_probability >= 0.75:

        risk_level = "HIGH"

    elif fake_probability >= 0.50:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    detection_coverage = (
        detection_count
        / NUM_FRAMES
    ) * 100.0

    return {

        "prediction":
            prediction,

        "authenticity_score":
            authenticity_score,

        "fake_probability":
            fake_probability_percent,

        "risk_level":
            risk_level,

        "frames_analyzed":
            NUM_FRAMES,

        "faces_detected":
            detection_count,

        "detection_coverage":
            detection_coverage,
    }


# ============================================================
# REPORT
# ============================================================

def generate_report(
    filename,
    metadata,
    results
):

    return f"""
DEEPGUARD
Deepfake Detection & Media Forensics Report
============================================================

FILE INFORMATION
------------------------------------------------------------
Filename:
{filename}

Resolution:
{metadata["width"]} × {metadata["height"]}

FPS:
{metadata["fps"]:.2f}

Original Frames:
{metadata["frame_count"]}

Duration:
{metadata["duration"]:.2f} seconds


DETECTION RESULT
------------------------------------------------------------
Prediction:
{results["prediction"]}

Authenticity Score:
{results["authenticity_score"]:.2f}%

Fake Probability:
{results["fake_probability"]:.2f}%

Risk Level:
{results["risk_level"]}


FORENSIC ANALYSIS
------------------------------------------------------------
Frames Analyzed:
{results["frames_analyzed"]}

Faces Detected:
{results["faces_detected"]}

Detection Coverage:
{results["detection_coverage"]:.2f}%


MODEL PIPELINE
------------------------------------------------------------
Video
  ↓
Frame Sampling
  ↓
Face Detection
  ↓
Face Alignment
  ↓
EfficientNet-B0
  ↓
1280-D Spatial Features
  ↓
Temporal Transformer
  ↓
Authenticity Prediction


MODEL CONFIGURATION
------------------------------------------------------------
Spatial Model:
EfficientNet-B0

Temporal Model:
Temporal Transformer

Frames:
64

Image Size:
224 × 224

Spatial Features:
1280

Transformer Hidden Size:
256

Transformer Heads:
8

Transformer Layers:
2


BENCHMARK
------------------------------------------------------------
Held-out Test Accuracy:
75.00%

Held-out Test Precision:
66.67%

Held-out Test Recall:
100.00%

Held-out Test F1:
80.00%


DISCLAIMER
------------------------------------------------------------
DeepGuard provides an AI-generated forensic estimate.
It should not be treated as definitive proof of
authenticity or manipulation.

Results may vary depending on video quality,
compression, lighting, pose, face visibility,
and manipulation technique.
""".strip()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🛡️ DeepGuard")

    st.write(
        "Deepfake Detection & Media Forensics"
    )

    st.divider()

    st.subheader("Model Benchmark")

    st.metric(
        "Test Accuracy",
        "75.00%"
    )

    st.metric(
        "Test F1",
        "80.00%"
    )

    st.metric(
        "Test Recall",
        "100.00%"
    )

    st.divider()

    st.subheader("Configuration")

    st.write(
        f"Frames analyzed: {NUM_FRAMES}"
    )

    st.write(
        f"Input size: "
        f"{IMAGE_SIZE} × {IMAGE_SIZE}"
    )

    st.write(
        "Spatial model: EfficientNet-B0"
    )

    st.write(
        "Temporal model: Transformer"
    )

    st.write(
        f"Device: {DEVICE}"
    )

    st.divider()

    st.caption(
        "DeepGuard provides probabilistic AI-assisted "
        "forensic analysis."
    )


# ============================================================
# UPLOAD
# ============================================================

st.header("📤 Upload Media")

uploaded_file = st.file_uploader(
    "Upload a video for analysis",
    type=[
        "mp4",
        "mov",
        "avi",
        "mkv",
        "webm",
    ],
)


# ============================================================
# IF VIDEO UPLOADED
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Temporary directory
    # --------------------------------------------------------

    temp_dir = "temp"

    os.makedirs(
        temp_dir,
        exist_ok=True
    )

    video_path = os.path.join(
        temp_dir,
        uploaded_file.name
    )

    # --------------------------------------------------------
    # Save video
    # --------------------------------------------------------

    with open(
        video_path,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )

    # --------------------------------------------------------
    # Video preview
    # --------------------------------------------------------

    st.header("🎥 Uploaded Video")

    st.video(
        uploaded_file
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    try:

        metadata = (
            get_video_metadata(
                video_path
            )
        )

    except Exception as error:

        st.error(
            f"Could not read video: {error}"
        )

        st.stop()

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    st.subheader(
        "📊 Video Information"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "Resolution",
            (
                f'{metadata["width"]} × '
                f'{metadata["height"]}'
            )
        )

    with col2:

        st.metric(
            "FPS",
            f'{metadata["fps"]:.0f}'
        )

    with col3:

        st.metric(
            "Duration",
            f'{metadata["duration"]:.2f}s'
        )

    with col4:

        st.metric(
            "Original Frames",
            metadata["frame_count"]
        )

    st.divider()

    # --------------------------------------------------------
    # Analyze button
    # --------------------------------------------------------

    analyze_button = st.button(
        "🔍 Analyze Video",
        type="primary",
        width="stretch",
    )

    if analyze_button:

        with st.spinner(
            "Analyzing video with DeepGuard..."
        ):

            try:

                results = analyze_video(
                    video_path
                )

            except Exception as error:

                st.error(
                    f"Analysis failed: {error}"
                )

                st.exception(
                    error
                )

                st.stop()

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.header(
            "🔎 Detection Result"
        )

        if results["prediction"] == "FAKE":

            st.error(
                "⚠️ FAKE DETECTED"
            )

            st.write(
                "DeepGuard detected a higher "
                "probability of synthetic manipulation."
            )

        else:

            st.success(
                "✅ REAL"
            )

            st.write(
                "DeepGuard detected a lower "
                "probability of synthetic manipulation."
            )

        # ----------------------------------------------------
        # Main scores
        # ----------------------------------------------------

        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:

            st.metric(
                "Authenticity Score",
                (
                    f'{results["authenticity_score"]:.2f}%'
                )
            )

        with col2:

            st.metric(
                "Fake Probability",
                (
                    f'{results["fake_probability"]:.2f}%'
                )
            )

        with col3:

            st.metric(
                "Risk Level",
                results["risk_level"]
            )

        # ----------------------------------------------------
        # Authenticity bar
        # ----------------------------------------------------

        st.write(
            "Authenticity Confidence"
        )

        st.progress(
            min(
                max(
                    results[
                        "authenticity_score"
                    ] / 100.0,
                    0.0
                ),
                1.0
            )
        )

        # ----------------------------------------------------
        # Forensic statistics
        # ----------------------------------------------------

        st.header(
            "📊 Forensic Statistics"
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:

            st.metric(
                "Frames Analyzed",
                results["frames_analyzed"]
            )

        with col2:

            st.metric(
                "Faces Detected",
                results["faces_detected"]
            )

        with col3:

            st.metric(
                "Detection Coverage",
                (
                    f'{results["detection_coverage"]:.0f}%'
                )
            )

        # ----------------------------------------------------
        # Pipeline
        # ----------------------------------------------------

        st.header(
            "🧠 Detection Pipeline"
        )

        pipeline_columns = (
            st.columns(7)
        )

        pipeline_steps = [

            "🎞️ Frames",

            "👤 Face Detection",

            "📐 Alignment",

            "🧩 EfficientNet",

            "📊 Features",

            "⏱️ Transformer",

            "🛡️ Prediction",
        ]

        for column, step in zip(
            pipeline_columns,
            pipeline_steps
        ):

            with column:

                st.info(step)

        # ----------------------------------------------------
        # Methodology
        # ----------------------------------------------------

        with st.expander(
            "📚 Model & Methodology"
        ):

            st.markdown(
                """
                ### Spatial Feature Extraction

                DeepGuard uses **EfficientNet-B0**
                to extract 1280-dimensional spatial
                features from aligned face images.

                ### Temporal Modeling

                Features from 64 sampled frames are
                passed to a **Temporal Transformer**.

                ### Face Processing

                The inference pipeline uses:

                - Haar Cascade face detection
                - Largest-face selection
                - Face margin = 0.2
                - 224 × 224 face processing
                - MediaPipe face alignment
                - BGR → RGB conversion
                - Pixel normalization to [0,1]

                ### Prediction

                The Transformer outputs a fake probability.

                Probability >= 50%:

                **FAKE**

                Probability < 50%:

                **REAL**
                """
            )

        # ----------------------------------------------------
        # Report
        # ----------------------------------------------------

        st.header(
            "📄 Forensic Report"
        )

        report = generate_report(
            uploaded_file.name,
            metadata,
            results
        )

        st.text_area(
            "Report",
            report,
            height=450
        )

        st.download_button(
            "⬇️ Download Forensic Report",
            data=report,
            file_name=(
                os.path.splitext(
                    uploaded_file.name
                )[0]
                + "_deepguard_report.txt"
            ),
            mime="text/plain",
            width="stretch",
        )

        # ----------------------------------------------------
        # Disclaimer
        # ----------------------------------------------------

        st.caption(
            "⚠️ DeepGuard provides an AI-generated "
            "forensic estimate. It should not be treated "
            "as definitive proof of authenticity or "
            "manipulation."
        )

else:

    # --------------------------------------------------------
    # Empty state
    # --------------------------------------------------------

    st.info(
        "Upload an MP4, MOV, AVI, MKV, or WebM "
        "video above to begin analysis."
    )

    st.header(
        "How DeepGuard Works"
    )

    st.markdown(
        """
        **1. 🎞️ Frame Sampling**

        64 frames are sampled from the video.

        **2. 👤 Face Detection**

        Faces are detected using OpenCV Haar Cascade.

        **3. 📐 Face Alignment**

        Facial landmarks are used to align the face.

        **4. 🧩 Spatial Analysis**

        EfficientNet-B0 extracts spatial facial features.

        **5. ⏱️ Temporal Analysis**

        A Transformer analyzes relationships between
        facial features across time.

        **6. 🛡️ Authenticity Prediction**

        The model produces a fake probability.

        **7. 📄 Forensic Report**

        DeepGuard generates a report containing the
        detection result and analysis statistics.
        """
    )