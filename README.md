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
