import torch

from src.models.efficientnet_feature_extractor import (
    EfficientNetFeatureExtractor,
)


def main():
    print("=== EfficientNet Feature Extraction Test ===")

    # Create the feature extractor.
    extractor = EfficientNetFeatureExtractor()

    print("EfficientNet loaded successfully.")

    # Create one dummy 224x224 RGB image.
    dummy_image = torch.randn(1, 3, 224, 224)

    print(f"Input shape: {dummy_image.shape}")

    # Extract features.
    features = extractor(dummy_image)

    print(f"Feature shape: {features.shape}")

    print("\n=== Test Summary ===")
    print("Model: EfficientNet-B0")
    print(f"Input: {dummy_image.shape}")
    print(f"Output: {features.shape}")


if __name__ == "__main__":
    main()