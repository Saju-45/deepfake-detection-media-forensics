# DeepGuard — Deepfake Detection & Media Forensics

> An AI-powered media forensics system that analyzes videos for signs of manipulation and produces an authenticity assessment with supporting forensic evidence.

DeepGuard is a deep-learning based deepfake detection system designed to go beyond a simple **REAL / FAKE** prediction.

The system analyzes facial content across multiple video frames, extracts spatial features using a pretrained EfficientNet model, models temporal information using a Transformer, and combines the analysis into an interpretable forensic report.

---

## 🚨 Why DeepGuard?

AI-generated and manipulated media is becoming increasingly difficult to distinguish from authentic content.

A single-frame classifier can identify visual artifacts, but deepfakes are fundamentally a **temporal problem**. Manipulations may appear inconsistently across frames through:

- facial artifacts
- unnatural expressions
- inconsistent facial features
- blending boundaries
- temporal instability
- frame-to-frame inconsistencies

This led to the central idea behind DeepGuard:

> **Don't ask whether one frame looks fake. Analyze how the face behaves across an entire sequence.**

DeepGuard therefore treats a video as a sequence rather than an isolated image.

---

# 🎯 Project Goal

Build an end-to-end AI system capable of:

1. Accepting a real-world video
2. Extracting representative frames
3. Detecting faces
4. Aligning facial regions
5. Extracting spatial visual features
6. Modeling temporal relationships between frames
7. Producing an authenticity score
8. Extracting supporting evidence
9. Generating a forensic analysis report

---

# 🧠 System Architecture

```text
                    ┌────────────────────┐
                    │    Input Video     │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │   Frame Sampling   │
                    │   8/16/32/64       │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │   Face Detection   │
                    │    Haar Cascade    │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │   Face Alignment   │
                    │     MediaPipe      │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Spatial Features   │
                    │   EfficientNet-B0  │
                    │     1280-D         │
                    └─────────┬──────────┘
                              │
                              ▼
              ┌──────────────────────────────┐
              │      Temporal Modeling       │
              │                              │
              │   Transformer Encoder        │
              │   64-frame sequence          │
              └──────────────┬───────────────┘
                             │
                             ▼
                    ┌────────────────────┐
                    │ Authenticity Score │
                    │   REAL / FAKE      │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Evidence Extraction│
                    │ & Forensic Report  │
                    └────────────────────┘
🔬 Technical Approach
1. Video Processing

Videos are sampled into a fixed number of frames.

Multiple sampling configurations were evaluated:

8 frames
16 frames
32 frames
64 frames

The purpose was to understand how temporal resolution affects detection performance.

2. Face Detection

Faces are detected independently in sampled frames using OpenCV's Haar Cascade face detector.

The detected facial region is then passed to the face-processing pipeline.

3. Face Alignment

Detected faces are aligned using MediaPipe Face Landmarker.

Eye landmarks are used to normalize the facial orientation before feature extraction.

This reduces variation caused by:

head rotation
face positioning
scale
alignment differences
4. Spatial Feature Extraction

DeepGuard uses a pretrained EfficientNet-B0 backbone.

The classification head is removed and the network is used as a feature extractor.

Each processed face is converted into a:

1280-dimensional feature vector

These features represent spatial information contained within the facial image.

The EfficientNet backbone is frozen during the temporal experiments.

5. Temporal Modeling

The sequence of frame-level feature vectors is passed to a Transformer Encoder.

The current best configuration uses:

Input features:       1280
Hidden dimension:      256
Attention heads:      8
Transformer layers:   2
Dropout:              0.3
Sequence length:      64 frames

The Transformer allows the model to learn relationships between facial features across time rather than evaluating frames independently.

🧪 Experiments

DeepGuard was developed through several controlled experiments.

Model comparison

Two temporal architectures were evaluated:

LSTM
Transformer Encoder
Frame sampling

The Transformer was evaluated with:

8 frames
16 frames
32 frames
64 frames

This helped investigate the relationship between temporal context and generalization on the held-out test set.

📊 Results

The dataset used for the current experiments contains:

Total videos:       106
Real videos:         53
Fake videos:         53

Training:            74
Validation:          16
Test:                16

The split is performed by video pair/index to maintain separation between training, validation, and test samples.

Frame Sampling Experiment
Frames	Validation Accuracy	Test Accuracy	Precision	Recall	F1
8	75.00%	68.75%	63.64%	87.50%	73.68%
16	75.00%	56.25%	57.14%	50.00%	53.33%
32	68.75%	68.75%	63.64%	87.50%	73.68%
64	81.25%	75.00%	66.67%	100.00%	80.00%
Best configuration

The 64-frame Transformer achieved:

75.00% test accuracy

with:

Precision: 66.67%
Recall: 100.00%
F1 Score: 80.00%

The model correctly detected all fake samples in the current held-out test set, while producing some false positives on real samples.

Important: These results come from a relatively small dataset and should not be interpreted as production-level accuracy. Larger and more diverse datasets are required to establish robust real-world generalization.

📈 What We Learned

The experiments revealed an important pattern:

More temporal context helped.

The 64-frame configuration performed better on the held-out test set than the 8-, 16-, and 32-frame configurations.

This supports the core design hypothesis that deepfake detection benefits from analyzing temporal behavior, rather than relying exclusively on individual frames.

However, the results also show that validation performance does not always translate directly into test performance.

This highlights an important challenge in deepfake detection:

Generalization is often harder than achieving high validation accuracy.

Future experiments will therefore focus on larger datasets, cross-dataset evaluation, stronger augmentation, and more robust temporal modeling.

🖥️ DeepGuard Dashboard

DeepGuard includes a Streamlit-based forensic analysis dashboard.

The application provides:

Video upload
Video metadata
Frame sampling
Face detection
Face alignment
Authenticity prediction
Fake probability
Risk level
Detection coverage
Evidence frames
Detection pipeline visualization
Model methodology
Downloadable forensic report

Example analysis:

Prediction:        REAL
Authenticity:      80.03%
Fake Probability:  19.97%
Risk Level:        LOW
Frames Analyzed:   64
Faces Detected:    64
Detection Coverage: 100%
🧾 Forensic Report

DeepGuard converts the model output into a human-readable forensic report.

The report summarizes:

Video metadata
Number of frames analyzed
Face detection coverage
Authenticity prediction
Authenticity score
Fake probability
Risk assessment
Evidence frames
Model methodology

The goal is to make the system useful not only as a classifier, but as a media-forensics tool.

🛠️ Tech Stack
Component	Technology
Language	Python
Deep Learning	PyTorch
Computer Vision	OpenCV
Face Alignment	MediaPipe
CNN Backbone	EfficientNet-B0
Temporal Model	Transformer Encoder
Data Processing	NumPy, Pandas
Evaluation	scikit-learn
Visualization	Matplotlib
UI	Streamlit
Development	VS Code
📁 Project Structure
deepfake-detection-media-forensics/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── experiment_frame_sampling.py
│   ├── test_temporal_transformer.py
│   └── test_transformer.py
│
├── src/
│   ├── preprocessing/
│   │   ├── video_processor.py
│   │   ├── face_detector.py
│   │   ├── face_processor.py
│   │   └── face_aligner.py
│   │
│   ├── models/
│   │   ├── efficientnet_feature_extractor.py
│   │   └── temporal_transformer.py
│   │
│   ├── training/
│   │   └── train_transformer.py
│   │
│   └── evaluation/
│
├── results/
│   └── models/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
🚀 Installation
1. Clone the repository
git clone https://github.com/Saju-45/deepfake-detection-media-forensics.git
cd deepfake-detection-media-forensics
2. Create a virtual environment
Windows
python -m venv .venv
.venv\Scripts\activate
Linux / macOS
python -m venv .venv
source .venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
📦 Dataset

The repository intentionally does not include the video dataset because of its size and licensing considerations.

Place your dataset locally under:

data/
├── raw/
│   ├── real/
│   └── fake/

The current experimental dataset contains:

53 real videos
53 fake videos

Dataset files are excluded from Git using .gitignore.

▶️ Running DeepGuard

Launch the Streamlit dashboard:

streamlit run app.py

Then open the local Streamlit URL shown in the terminal.

Upload a video and DeepGuard will run the complete pipeline:

Video
 ↓
Frame Sampling
 ↓
Face Detection
 ↓
Face Alignment
 ↓
EfficientNet Features
 ↓
Temporal Transformer
 ↓
Authenticity Prediction
 ↓
Forensic Evidence
 ↓
Report
💡 Future Work

DeepGuard is an actively evolving research project.

Planned improvements include:

Better datasets

Evaluate on larger and more diverse datasets such as:

FaceForensics++
Celeb-DF
DFDC
DeeperForensics-1.0
Cross-dataset generalization

Train on one dataset and evaluate on another to measure real-world robustness.

Stronger temporal modeling

Investigate:

longer temporal sequences
temporal attention
Temporal Transformers
CNN + Transformer hybrids
Better spatial representations

Evaluate:

stronger EfficientNet variants
Vision Transformers
facial-region specific feature extraction
Explainability

Add:

Grad-CAM
Transformer attention visualization
suspicious-frame ranking
artifact localization
Robustness

Evaluate performance under:

compression
resizing
low resolution
lighting changes
frame dropping
social-media transcoding
⚠️ Limitations

The current system is a research prototype rather than a production forensic authority.

Important limitations include:

relatively small experimental dataset
limited test-set size
possible dataset-specific artifacts
no guarantee of generalization to unseen generation methods
Haar Cascade face detection can fail under difficult conditions
current model evaluation is not yet cross-dataset
CPU inference limits experimentation speed

A high confidence score should therefore not be interpreted as definitive proof that media is authentic or manipulated.

🔐 Responsible AI

Deepfake detection is a high-impact application.

False positives can incorrectly label authentic media as manipulated, while false negatives can allow manipulated media to appear authentic.

DeepGuard is therefore designed as a decision-support and forensic analysis tool, not as an unquestionable source of truth.

The project prioritizes:

transparent evaluation
evidence-based predictions
explicit limitations
reproducible experiments
responsible interpretation of model confidence
👨‍💻 Project

DeepGuard — Deepfake Detection & Media Forensics

Built as an AI deep-learning project exploring spatial + temporal modeling for manipulated media detection.

The project focuses on the intersection of:

Computer Vision
+
Deep Learning
+
Video Understanding
+
Temporal Modeling
+
Media Forensics
