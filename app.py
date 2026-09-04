import os
import tempfile
import textwrap
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
import torch

from src.preprocessing.video_processor import VideoProcessor
from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.face_processor import FaceProcessor
from src.preprocessing.face_aligner import FaceAligner
from src.models.efficientnet_feature_extractor import EfficientNetFeatureExtractor
from src.models.temporal_transformer import TemporalTransformer


# ============================================================
# HTML RENDER HELPER
# ============================================================

def render_html(content):
    """Render custom HTML without Markdown treating indented HTML as code."""
    content = textwrap.dedent(content).strip()

    if hasattr(st, "html"):
        st.html(content)
    else:
        st.markdown(content, unsafe_allow_html=True)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DeepGuard | Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH = "results/models/frame_sampling_transformer.pt"

NUM_FRAMES = 64
IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CUSTOM CSS
# ============================================================

render_html(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(99, 102, 241, 0.10),
                transparent 35%
            ),
            radial-gradient(
                circle at top left,
                rgba(14, 165, 233, 0.08),
                transparent 30%
            );
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    .hero {
        padding: 1.8rem 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.035);
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2.7rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #9ca3af;
        max-width: 850px;
    }

    .metric-card {
        border-radius: 16px;
        padding: 1.15rem;
        border: 1px solid rgba(255,255,255,0.09);
        background: rgba(255,255,255,0.035);
        min-height: 115px;
    }

    .metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.65rem;
        font-weight: 750;
    }

    .result-card {
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.04);
        margin: 1rem 0;
    }

    .result-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }

    .result-description {
        color: #9ca3af;
        font-size: 1rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 750;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    .pipeline {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 1rem 0;
    }

    .pipeline-step {
        padding: 8px 12px;
        border-radius: 10px;
        background: rgba(255,255,255,0.055);
        border: 1px solid rgba(255,255,255,0.08);
        font-size: 0.84rem;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    [data-testid="stFileUploader"] {
        border-radius: 16px;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 44px;
    }

    textarea {
        font-family:
            "SFMono-Regular",
            Consolas,
            "Liberation Mono",
            monospace !important;
        font-size: 0.82rem !important;
    }

    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.07);
    }

    </style>
    """
)


# ============================================================
# HERO HEADER
# ============================================================

render_html(
    """
    <div class="hero">

        <div class="hero-title">
            🛡️ DeepGuard
        </div>

        <div class="hero-subtitle">
            Deepfake Detection & Media Forensics using
            spatial deep-learning features and temporal analysis.
        </div>

    </div>
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🛡️ DeepGuard")

    st.caption(
        "AI-powered video authenticity analysis"
    )

    st.markdown("---")

    st.markdown("### 🔬 Detection Pipeline")

    st.markdown(
        """
        **1.** Video frame extraction

        **2.** Face detection

        **3.** Face preprocessing

        **4.** Face alignment

        **5.** EfficientNet spatial features

        **6.** Temporal Transformer

        **7.** Authenticity scoring

        **8.** Forensic evidence report
        """
    )

    st.markdown("---")

    st.markdown("### 🤖 Current Model")

    st.write("**Spatial backbone:** EfficientNet-B0")
    st.write("**Temporal model:** Transformer")
    st.write("**Frames:** 64")
    st.write("**Feature size:** 1280")
    st.write("**Input:** 224 × 224")
    st.write(f"**Device:** {DEVICE}")

    st.markdown("---")

    st.markdown("### 📊 Current Benchmark")

    st.metric(
        "Test Accuracy",
        "75.00%"
    )

    st.metric(
        "Test F1",
        "80.00%"
    )

    st.metric(
        "Recall",
        "100.00%"
    )

    st.caption(
        "Current 64-frame experiment benchmark."
    )


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    # EfficientNet-B0 feature extractor
    feature_extractor = EfficientNetFeatureExtractor()

    feature_extractor = feature_extractor.to(DEVICE)
    feature_extractor.eval()

    # Temporal Transformer
    transformer = TemporalTransformer(
        input_size=1280,
        hidden_size=256,
        num_heads=8,
        num_layers=2,
        dropout=0.3,
        max_frames=64,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    transformer.load_state_dict(
        checkpoint
    )

    transformer = transformer.to(DEVICE)
    transformer.eval()

    return (
        feature_extractor,
        transformer,
    )


# ============================================================
# VIDEO METADATA
# ============================================================

def get_video_metadata(video_path):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        return {
            "fps": 0,
            "frames": 0,
            "width": 0,
            "height": 0,
            "duration": 0,
        }

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

    duration = (
        frame_count / fps
        if fps > 0
        else 0
    )

    cap.release()

    return {
        "fps": fps,
        "frames": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
    }


# ============================================================
# PREPARE VIDEO
# ============================================================

def prepare_video(video_path):

    video_processor = VideoProcessor(
        video_path
    )

    face_detector = FaceDetector()

    face_processor = FaceProcessor()

    face_aligner = FaceAligner()

    try:

        # ----------------------------------------------------
        # SAMPLE 64 FRAMES
        # ----------------------------------------------------

        frames = video_processor.sample_frames(
            num_frames=NUM_FRAMES
        )

        processed_faces = []

        evidence_frames = []

        detected_count = 0

        # ----------------------------------------------------
        # PROCESS EACH FRAME
        # ----------------------------------------------------

        for index, frame in enumerate(frames):

            # Detect faces
            faces = face_detector.detect(
                frame
            )

            if faces is None or len(faces) == 0:
                continue

            # ------------------------------------------------
            # SELECT LARGEST FACE
            # ------------------------------------------------

            largest_face = max(
                faces,
                key=lambda box: box[2] * box[3]
            )

            # ------------------------------------------------
            # PROCESS FACE
            #
            # FaceProcessor.process_frame() expects a LIST
            # of bounding boxes and returns a LIST of faces.
            # ------------------------------------------------

            processed_face_list = (
                face_processor.process_frame(
                    frame,
                    [largest_face]
                )
            )

            if not processed_face_list:
                continue

            # Extract first processed face
            face = processed_face_list[0]

            if face is None or face.size == 0:
                continue

            # ------------------------------------------------
            # ALIGN FACE
            # ------------------------------------------------

            aligned = face_aligner.align(
                face
            )

            # Fallback to processed crop
            if aligned is None:

                aligned = face

            if aligned is None or aligned.size == 0:
                continue

            # ------------------------------------------------
            # BGR → RGB
            # ------------------------------------------------

            aligned_rgb = cv2.cvtColor(
                aligned,
                cv2.COLOR_BGR2RGB
            )

            processed_faces.append(
                aligned_rgb
            )

            detected_count += 1

            # ------------------------------------------------
            # VISUAL EVIDENCE
            # ------------------------------------------------

            if len(evidence_frames) < 6:

                evidence_frames.append(
                    aligned_rgb
                )

        # ----------------------------------------------------
        # NO FACE FOUND
        # ----------------------------------------------------

        if len(processed_faces) == 0:

            return (
                None,
                0,
                []
            )

        # ----------------------------------------------------
        # REPEAT LAST VALID FACE
        # UNTIL 64 FRAMES
        # ----------------------------------------------------

        while len(processed_faces) < NUM_FRAMES:

            processed_faces.append(
                processed_faces[-1].copy()
            )

        # Exactly 64 frames
        processed_faces = (
            processed_faces[:NUM_FRAMES]
        )

        # ----------------------------------------------------
        # NUMPY ARRAY
        # ----------------------------------------------------

        faces_array = np.stack(
            processed_faces
        )

        # ----------------------------------------------------
        # NUMPY → TORCH
        # ----------------------------------------------------

        faces_tensor = torch.from_numpy(
            faces_array
        ).permute(
            0,
            3,
            1,
            2
        ).float()

        return (
            faces_tensor,
            detected_count,
            evidence_frames,
        )

    finally:

        try:
            face_aligner.close()
        except Exception:
            pass


# ============================================================
# ANALYZE VIDEO
# ============================================================

def analyze_video(video_path):

    feature_extractor, transformer = (
        load_models()
    )

    # --------------------------------------------------------
    # PREPARE VIDEO
    # --------------------------------------------------------

    (
        faces_tensor,
        detected_count,
        evidence_frames,
    ) = prepare_video(
        video_path
    )

    if faces_tensor is None:

        return {
            "error":
            "No usable face was detected "
            "in the uploaded video."
        }

    # --------------------------------------------------------
    # MOVE TO DEVICE
    # --------------------------------------------------------

    faces_tensor = faces_tensor.to(
        DEVICE
    )

    # --------------------------------------------------------
    # EFFICIENTNET PREPROCESSING
    # --------------------------------------------------------

    processed = []

    for face in faces_tensor:

        processed_face = (
            feature_extractor.preprocess(
                face
            )
        )

        processed.append(
            processed_face
        )

    processed = torch.stack(
        processed
    )

    # --------------------------------------------------------
    # SPATIAL FEATURES
    # --------------------------------------------------------

    with torch.no_grad():

        spatial_features = (
            feature_extractor(
                processed
            )
        )

        # [64, 1280]
        #
        # →
        #
        # [1, 64, 1280]

        temporal_input = (
            spatial_features.unsqueeze(0)
        )

        # ----------------------------------------------------
        # TEMPORAL TRANSFORMER
        # ----------------------------------------------------

        logits = transformer(
            temporal_input
        )

        # ----------------------------------------------------
        # FAKE PROBABILITY
        # ----------------------------------------------------

        fake_probability = (
            torch.sigmoid(
                logits
            ).item()
        )

    # --------------------------------------------------------
    # SCORES
    # --------------------------------------------------------

    authenticity = (
        1.0 - fake_probability
    ) * 100

    fake_percentage = (
        fake_probability * 100
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if fake_probability >= 0.5:

        prediction = "FAKE"

    else:

        prediction = "REAL"

    # --------------------------------------------------------
    # RISK INDICATOR
    # --------------------------------------------------------

    if fake_percentage >= 75:

        risk_level = "HIGH"

    elif fake_percentage >= 50:

        risk_level = "MEDIUM"

    elif fake_percentage >= 25:

        risk_level = "LOW-MEDIUM"

    else:

        risk_level = "LOW"

    # --------------------------------------------------------
    # FACE COVERAGE
    # --------------------------------------------------------

    coverage = (
        detected_count /
        NUM_FRAMES
    ) * 100

    return {

        "prediction":
            prediction,

        "fake_probability":
            fake_probability,

        "fake_percentage":
            fake_percentage,

        "authenticity":
            authenticity,

        "risk_level":
            risk_level,

        "detected_count":
            detected_count,

        "coverage":
            coverage,

        "evidence_frames":
            evidence_frames,

        "frames_analyzed":
            NUM_FRAMES,
    }


# ============================================================
# FORENSIC REPORT GENERATOR
# ============================================================

def generate_report(
    prediction,
    fake_probability,
    authenticity,
    risk_level,
    detected_count,
    total_frames,
    metadata,
):

    fake_percentage = (
        fake_probability * 100
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    coverage = (
        detected_count /
        total_frames
    ) * 100

    report = f"""
============================================================
DEEPGUARD - FORENSIC VIDEO ANALYSIS REPORT
============================================================

Generated:
{timestamp}

------------------------------------------------------------
ANALYSIS RESULT
------------------------------------------------------------

Prediction:
{prediction}

Authenticity Score:
{authenticity:.2f}%

Fake Probability:
{fake_percentage:.2f}%

Model Risk Indicator:
{risk_level}

------------------------------------------------------------
VIDEO INFORMATION
------------------------------------------------------------

Resolution:
{metadata["width"]} × {metadata["height"]}

FPS:
{metadata["fps"]:.2f}

Original Frames:
{metadata["frames"]}

Duration:
{metadata["duration"]:.2f} seconds

------------------------------------------------------------
FORENSIC ANALYSIS
------------------------------------------------------------

Frames Analyzed:
{total_frames}

Faces Successfully Detected:
{detected_count}

Face Detection Coverage:
{coverage:.2f}%

------------------------------------------------------------
MODEL PIPELINE
------------------------------------------------------------

Video
  ↓
Frame Sampling
  ↓
Face Detection
  ↓
Face Preprocessing
  ↓
Face Alignment
  ↓
EfficientNet-B0 Spatial Features
  ↓
Temporal Transformer
  ↓
Authenticity Probability
  ↓
Forensic Report

------------------------------------------------------------
MODEL CONFIGURATION
------------------------------------------------------------

Spatial Backbone:
EfficientNet-B0

Spatial Feature Dimension:
1280

Temporal Model:
Temporal Transformer

Temporal Frames:
64

Input Face Resolution:
224 × 224

Inference Device:
{DEVICE}

------------------------------------------------------------
FRAME SAMPLING EXPERIMENT
------------------------------------------------------------

Frames     Validation     Test Accuracy     Precision     Recall     F1
8          75.00%         68.75%             63.64%        87.50%    73.68%
16         75.00%         56.25%             57.14%        50.00%    53.33%
32         68.75%         68.75%             63.64%        87.50%    73.68%
64         81.25%         75.00%             66.67%        100.00%   80.00%

Selected Configuration:
64 frames

------------------------------------------------------------
INTERPRETATION
------------------------------------------------------------

The prediction is produced by a deep-learning pipeline that
combines frame-level facial features with temporal modeling.

The authenticity score represents the model's estimated
confidence that the analyzed video is authentic.

The fake probability represents the model's estimated
probability that the video contains manipulated content.

The model risk indicator is a presentation-level threshold
based on the predicted fake probability and should not be
interpreted as a calibrated probability of real-world harm.

------------------------------------------------------------
IMPORTANT DISCLAIMER
------------------------------------------------------------

DeepGuard is an experimental AI-based media forensics system.

The output should be treated as an automated screening result,
not as definitive proof of authenticity or manipulation.

Performance can vary across datasets, compression levels,
video quality, identities, lighting conditions, and
manipulation methods.

============================================================
END OF REPORT
============================================================
"""

    return report.strip()


# ============================================================
# UPLOAD SECTION
# ============================================================

render_html(
    '<div class="section-title">🎥 Upload Media</div>'
)

uploaded_file = st.file_uploader(
    "Upload a video for deepfake analysis",
    type=[
        "mp4",
        "mov",
        "avi",
        "mkv",
    ],
    help=(
        "Supported video formats: MP4, MOV, AVI and MKV."
    ),
)


# ============================================================
# NO FILE
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload a video above to begin forensic analysis."
    )

    st.markdown(
        """
        ### What DeepGuard analyzes

        - Facial manipulation patterns
        - Frame-level spatial features
        - Temporal consistency
        - Face detection coverage
        - Authenticity probability
        - Automated forensic evidence

        **Recommended:** use a short video containing a
        clearly visible face for the fastest demonstration.
        """
    )


# ============================================================
# FILE UPLOADED
# ============================================================

else:

    # --------------------------------------------------------
    # SAVE UPLOAD
    # --------------------------------------------------------

    suffix = os.path.splitext(
        uploaded_file.name
    )[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as tmp:

        tmp.write(
            uploaded_file.read()
        )

        video_path = tmp.name

    # --------------------------------------------------------
    # PREVIEW
    # --------------------------------------------------------

    render_html(
        '<div class="section-title">🎬 Media Preview</div>'
    )

    col_preview, col_info = st.columns(
        [1.6, 1]
    )

    metadata = get_video_metadata(
        video_path
    )

    with col_preview:

        st.video(
            video_path
        )

    with col_info:

        st.markdown(
            "### Video Metadata"
        )

        m1, m2 = st.columns(2)

        with m1:

            render_html(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        Resolution
                    </div>

                    <div class="metric-value">
                        {metadata["width"]}
                        ×
                        {metadata["height"]}
                    </div>

                </div>
                """
            )

        with m2:

            render_html(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        FPS
                    </div>

                    <div class="metric-value">
                        {metadata["fps"]:.1f}
                    </div>

                </div>
                """
            )

        st.write("")

        m3, m4 = st.columns(2)

        with m3:

            render_html(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        Frames
                    </div>

                    <div class="metric-value">
                        {metadata["frames"]}
                    </div>

                </div>
                """
            )

        with m4:

            render_html(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        Duration
                    </div>

                    <div class="metric-value">
                        {metadata["duration"]:.2f}s
                    </div>

                </div>
                """
            )

    st.markdown("")

    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    analyze_button = st.button(
        "🔍 Analyze Video",
        type="primary",
        width="stretch",
    )

    if analyze_button:

        with st.spinner(
            "Running DeepGuard forensic analysis..."
        ):

            try:

                results = analyze_video(
                    video_path
                )

            except Exception as e:

                st.error(
                    "An error occurred during analysis."
                )

                st.exception(e)

                results = None

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        if results is not None:

            if "error" in results:

                st.error(
                    results["error"]
                )

            else:

                st.markdown("---")

                render_html(
                    '<div class="section-title">🧠 Analysis Result</div>'
                )

                prediction = results[
                    "prediction"
                ]

                authenticity = results[
                    "authenticity"
                ]

                fake_percentage = results[
                    "fake_percentage"
                ]

                risk_level = results[
                    "risk_level"
                ]

                detected_count = results[
                    "detected_count"
                ]

                coverage = results[
                    "coverage"
                ]

                # ------------------------------------------------
                # RESULT MESSAGE
                # ------------------------------------------------

                if prediction == "REAL":

                    title = (
                        "Likely Authentic"
                    )

                    description = (
                        "The model estimates a lower "
                        "probability of facial manipulation "
                        "in the analyzed video."
                    )

                else:

                    title = (
                        "Potentially Manipulated"
                    )

                    description = (
                        "The model estimates a higher "
                        "probability of manipulation "
                        "in the analyzed video."
                    )

                render_html(
                    f"""
                    <div class="result-card">

                        <div class="result-title">
                            {title}
                        </div>

                        <div class="result-description">
                            {description}
                        </div>

                    </div>
                    """
                )

                # ------------------------------------------------
                # SCORE CARDS
                # ------------------------------------------------

                c1, c2, c3, c4 = st.columns(4)

                with c1:

                    render_html(
                        f"""
                        <div class="metric-card">

                            <div class="metric-label">
                                Prediction
                            </div>

                            <div class="metric-value">
                                {prediction}
                            </div>

                        </div>
                        """
                    )

                with c2:

                    render_html(
                        f"""
                        <div class="metric-card">

                            <div class="metric-label">
                                Authenticity
                            </div>

                            <div class="metric-value">
                                {authenticity:.2f}%
                            </div>

                        </div>
                        """
                    )

                with c3:

                    render_html(
                        f"""
                        <div class="metric-card">

                            <div class="metric-label">
                                Fake Probability
                            </div>

                            <div class="metric-value">
                                {fake_percentage:.2f}%
                            </div>

                        </div>
                        """
                    )

                with c4:

                    render_html(
                        f"""
                        <div class="metric-card">

                            <div class="metric-label">
                                Risk Indicator
                            </div>

                            <div class="metric-value">
                                {risk_level}
                            </div>

                        </div>
                        """
                    )

                # ------------------------------------------------
                # AUTHENTICITY BAR
                # ------------------------------------------------

                render_html(
                    '<div class="section-title">📈 Authenticity Analysis</div>'
                )

                st.progress(
                    min(
                        max(
                            authenticity / 100,
                            0.0
                        ),
                        1.0
                    )
                )

                col_a, col_b = st.columns(2)

                with col_a:

                    st.caption(
                        f"Authenticity estimate: "
                        f"{authenticity:.2f}%"
                    )

                with col_b:

                    st.caption(
                        f"Manipulation estimate: "
                        f"{fake_percentage:.2f}%"
                    )

                # ------------------------------------------------
                # FORENSIC EVIDENCE
                # ------------------------------------------------

                render_html(
                    '<div class="section-title">🔬 Forensic Evidence</div>'
                )

                e1, e2, e3 = st.columns(3)

                with e1:

                    render_html(
                        f"""
                        <div class="metric-card">

                            <div class="metric-label">
                                Frames Analyzed
                            </div>

                            <div class="metric-value">
                                {NUM_FRAMES}
                            </div>

                        </div>
                        """
                    )

                with e2:

                    render_html(
                        f"""
                        <div class="metric-card">

                            <div class="metric-label">
                                Faces Detected
                            </div>

                            <div class="metric-value">
                                {detected_count}
                            </div>

                        </div>
                        """
                    )

                with e3:

                    render_html(
                        f"""
                        <div class="metric-card">

                            <div class="metric-label">
                                Detection Coverage
                            </div>

                            <div class="metric-value">
                                {coverage:.1f}%
                            </div>

                        </div>
                        """
                    )

                # ------------------------------------------------
                # FACE EVIDENCE
                # ------------------------------------------------

                evidence_frames = results[
                    "evidence_frames"
                ]

                if evidence_frames:

                    render_html(
                        '<div class="section-title">👤 Sampled Facial Evidence</div>'
                    )

                    cols = st.columns(
                        len(evidence_frames)
                    )

                    for i, frame in enumerate(
                        evidence_frames
                    ):

                        with cols[i]:

                            st.image(
                                frame,
                                caption=(
                                    f"Evidence frame "
                                    f"{i + 1}"
                                ),
                                width="stretch",
                            )

                # ------------------------------------------------
                # DETECTION PIPELINE
                # ------------------------------------------------

                render_html(
                    '<div class="section-title">⚙️ Detection Pipeline</div>'
                )

                render_html(
                    """
                    <div class="pipeline">

                        <div class="pipeline-step">
                            🎥 Video
                        </div>

                        <div class="pipeline-step">
                            ↓ Frame Sampling
                        </div>

                        <div class="pipeline-step">
                            👤 Face Detection
                        </div>

                        <div class="pipeline-step">
                            ✂️ Face Processing
                        </div>

                        <div class="pipeline-step">
                            📐 Alignment
                        </div>

                        <div class="pipeline-step">
                            🧠 EfficientNet-B0
                        </div>

                        <div class="pipeline-step">
                            ⏱️ Temporal Transformer
                        </div>

                        <div class="pipeline-step">
                            📊 Authenticity Score
                        </div>

                        <div class="pipeline-step">
                            📄 Forensic Report
                        </div>

                    </div>
                    """
                )

                # ------------------------------------------------
                # EXPERIMENTAL RESULTS
                # ------------------------------------------------

                render_html(
                    '<div class="section-title">🧪 Frame Sampling Experiment</div>'
                )

                st.caption(
                    "Comparison of temporal frame counts "
                    "evaluated during model experimentation."
                )

                experiment_data = {

                    "Frames": [
                        8,
                        16,
                        32,
                        64,
                    ],

                    "Validation Accuracy": [
                        "75.00%",
                        "75.00%",
                        "68.75%",
                        "81.25%",
                    ],

                    "Test Accuracy": [
                        "68.75%",
                        "56.25%",
                        "68.75%",
                        "75.00%",
                    ],

                    "Precision": [
                        "63.64%",
                        "57.14%",
                        "63.64%",
                        "66.67%",
                    ],

                    "Recall": [
                        "87.50%",
                        "50.00%",
                        "87.50%",
                        "100.00%",
                    ],

                    "F1 Score": [
                        "73.68%",
                        "53.33%",
                        "73.68%",
                        "80.00%",
                    ],
                }

                st.dataframe(
                    experiment_data,
                    width="stretch",
                    hide_index=True,
                )

                st.success(
                    "64 frames selected as the current "
                    "configuration, with the highest "
                    "validation accuracy of 81.25%."
                )

                st.caption(
                    "The benchmark uses a small held-out "
                    "test split, so these metrics should "
                    "be interpreted as experimental "
                    "benchmarks rather than production-level "
                    "performance guarantees."
                )

                # ------------------------------------------------
                # MODEL METHODOLOGY
                # ------------------------------------------------

                with st.expander(
                    "🧠 Model & Methodology"
                ):

                    st.markdown(
                        """
                        ### Spatial Feature Extraction

                        Each detected and aligned face is
                        resized to **224 × 224 pixels** and
                        passed through a pretrained
                        **EfficientNet-B0** backbone.

                        The classification head is removed
                        and the network is used as a feature
                        extractor, producing a
                        **1280-dimensional spatial
                        representation** for every frame.

                        ### Temporal Modeling

                        The sequence of frame-level features
                        is passed into a **Temporal Transformer**.

                        Current configuration:

                        - Input dimension: 1280
                        - Hidden dimension: 256
                        - Attention heads: 8
                        - Transformer layers: 2
                        - Dropout: 0.3
                        - Maximum sequence length: 64

                        ### Decision

                        The Transformer produces a manipulation
                        logit. A sigmoid converts this value
                        into a fake probability.
                        """
                    )

                # ------------------------------------------------
                # FORENSIC REPORT
                # ------------------------------------------------

                st.markdown("---")

                render_html(
                    '<div class="section-title">📄 Forensic Report</div>'
                )

                report_text = generate_report(
                    prediction=prediction,
                    fake_probability=results[
                        "fake_probability"
                    ],
                    authenticity=authenticity,
                    risk_level=risk_level,
                    detected_count=detected_count,
                    total_frames=NUM_FRAMES,
                    metadata=metadata,
                )

                render_html(
                    """
                    <div class="result-card">

                        <b>Investigation Summary</b>

                        <br><br>

                        <span style="color:#9ca3af;">
                        The report below can be copied
                        directly or downloaded as a text
                        file for documentation, review,
                        or submission.
                        </span>

                    </div>
                    """
                )

                # ------------------------------------------------
                # COPYABLE REPORT
                # ------------------------------------------------

                st.text_area(
                    "📋 Copyable Report",
                    value=report_text,
                    height=430,
                    key="forensic_report",
                )

                # ------------------------------------------------
                # DOWNLOAD REPORT
                # ------------------------------------------------

                st.download_button(
                    label="⬇️ Download Forensic Report",
                    data=report_text,
                    file_name=(
                        "deepguard_forensic_report.txt"
                    ),
                    mime="text/plain",
                    width="stretch",
                )

                st.caption(
                    "To copy: click inside the report box, "
                    "press Ctrl+A, then Ctrl+C."
                )

                # ------------------------------------------------
                # DISCLAIMER
                # ------------------------------------------------

                st.markdown("---")

                st.warning(
                    "DeepGuard is an experimental AI-based "
                    "media forensics system. Its output "
                    "should be treated as an automated "
                    "screening result rather than definitive "
                    "proof of authenticity or manipulation."
                )


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div class="footer">

        <b>DeepGuard</b>
        · Deepfake Detection & Media Forensics

        <br>

        EfficientNet-B0 + Temporal Transformer

        <br><br>

        Built for AI-powered media authenticity analysis.

    </div>
    """
)