import torch

from src.training.video_dataset import VideoDataset
from src.models.efficientnet_feature_extractor import (
    EfficientNetFeatureExtractor,
)
from src.models.temporal_lstm import TemporalLSTM


def main():
    print("=== Full Deepfake Detection Pipeline Test ===\n")

    # --------------------------------------------------
    # 1. Load one video from the training dataset
    # --------------------------------------------------
    dataset = VideoDataset(
        root_dir="data/split/train",
        num_frames=16,
        image_size=(224, 224),
    )

    print(f"Dataset size: {len(dataset)}")

    video, label = dataset[0]

    print(f"Video tensor: {video.shape}")
    print(f"Label: {label.item()}")

    # Add batch dimension
    video = video.unsqueeze(0)

    print(f"Batch input: {video.shape}")

    # --------------------------------------------------
    # 2. Load EfficientNet
    # --------------------------------------------------
    spatial_model = EfficientNetFeatureExtractor()

    print("\nEfficientNet-B0 loaded.")

    # --------------------------------------------------
    # 3. Extract spatial features
    # --------------------------------------------------
    batch_size, num_frames, channels, height, width = video.shape

    # Convert:
    # [1, 16, 3, 224, 224]
    #
    # into:
    # [16, 3, 224, 224]
    #
    # so EfficientNet processes each frame.
    frames = video.view(
        batch_size * num_frames,
        channels,
        height,
        width,
    )

    with torch.no_grad():
        spatial_features = spatial_model(frames)

    print(f"Spatial features: {spatial_features.shape}")

    # Reshape:
    # [16, 1280]
    #
    # into:
    # [1, 16, 1280]
    temporal_input = spatial_features.view(
        batch_size,
        num_frames,
        -1,
    )

    print(f"Temporal input: {temporal_input.shape}")

    # --------------------------------------------------
    # 4. Temporal LSTM
    # --------------------------------------------------
    temporal_model = TemporalLSTM(
        input_size=1280,
        hidden_size=256,
        num_layers=2,
        dropout=0.3,
    )

    print("Temporal LSTM loaded.")

    logits = temporal_model(temporal_input)

    print(f"LSTM output: {logits.shape}")
    print(f"Raw logit: {logits.item():.4f}")

    # --------------------------------------------------
    # 5. Convert logit to probability
    # --------------------------------------------------
    probability = torch.sigmoid(logits)

    print(f"Fake probability: {probability.item():.4f}")

    # --------------------------------------------------
    # 6. Final prediction
    # --------------------------------------------------
    prediction = 1 if probability.item() >= 0.5 else 0

    prediction_name = (
        "FAKE" if prediction == 1 else "REAL"
    )

    print(f"Prediction: {prediction_name}")

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    print("\n=== Pipeline Summary ===")
    print("Video")
    print("  ↓")
    print("Frame sampling")
    print("  ↓")
    print("Face detection")
    print("  ↓")
    print("Face alignment")
    print("  ↓")
    print("EfficientNet-B0")
    print("  ↓")
    print("1280-D spatial features")
    print("  ↓")
    print("Temporal LSTM")
    print("  ↓")
    print("Fake probability")
    print("  ↓")
    print(f"{prediction_name}")

    dataset.close()


if __name__ == "__main__":
    main()