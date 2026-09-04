import torch

from src.models.temporal_lstm import TemporalLSTM


def main():
    print("=== Temporal LSTM Test ===")

    # Load features extracted from the video.
    features = torch.load(
        "results/features.pt",
        weights_only=True,
    )

    print(f"Loaded feature shape: {features.shape}")

    # Add batch dimension.
    features = features.unsqueeze(0)

    print(f"Input shape: {features.shape}")

    # Create LSTM model.
    model = TemporalLSTM()

    print("Temporal LSTM loaded successfully.")

    # Run temporal model.
    logits = model(features)

    print(f"Output shape: {logits.shape}")
    print(f"Output value: {logits.item():.4f}")

    print("\n=== Test Summary ===")
    print("Input: 16 frames × 1280 features")
    print(f"Batch input: {features.shape}")
    print(f"LSTM output: {logits.shape}")
    print("Temporal model test completed successfully.")


if __name__ == "__main__":
    main()