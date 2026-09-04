# DeepGuard — Deepfake Detection & Media Forensics

> **Don't just ask whether a frame looks fake. Analyze how the face behaves across an entire video sequence.**

DeepGuard is an AI-powered media forensics system that analyzes facial content across video frames to detect potential deepfake manipulation and generate an evidence-backed authenticity assessment.

Unlike a simple frame-level classifier, DeepGuard combines **spatial visual features** from EfficientNet-B0 with **temporal modeling** through a Transformer Encoder to capture inconsistencies that emerge across time.

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
## 📊 Results

DeepGuard was evaluated on a balanced dataset containing 106 videos:

- 53 real
- 53 fake
- 74 training videos
- 16 validation videos
- 16 test videos

The best-performing configuration used 64 sampled frames with an EfficientNet-B0 spatial feature extractor and Transformer temporal encoder.

| Frames | Validation Accuracy | Test Accuracy | Precision | Recall | F1 |
|------:|--------------------:|--------------:|----------:|-------:|---:|
| 8     | 75.00% | 68.75% | 63.64% | 87.50% | 73.68% |
| 16    | 75.00% | 56.25% | 57.14% | 50.00% | 53.33% |
| 32    | 68.75% | 68.75% | 63.64% | 87.50% | 73.68% |
| **64** | **81.25%** | **75.00%** | **66.67%** | **100.00%** | **80.00%** |

### Best Test Result

**75.00% accuracy · 66.67% precision · 100.00% recall · 80.00% F1**

The 64-frame configuration achieved the strongest performance on the held-out test set, supporting the hypothesis that additional temporal context can improve deepfake detection.

> These results are from a relatively small experimental dataset and should not be interpreted as production-level accuracy. Larger datasets and cross-dataset evaluation are required to establish real-world generalization.
