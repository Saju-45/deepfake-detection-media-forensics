
import torch

from src.models.temporal_transformer import TemporalTransformer


def main():

    print("=== Temporal Transformer Test ===")

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = TemporalTransformer()

    model.eval()

    print("Temporal Transformer loaded successfully.")

    # --------------------------------------------------
    # Create dummy EfficientNet feature sequence
    #
    # Shape:
    # [batch, frames, features]
    #
    # [1, 16, 1280]
    # --------------------------------------------------

    input_tensor = torch.randn(
        1,
        16,
        1280,
    )

    print(
        f"Input shape: {input_tensor.shape}"
    )

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    with torch.no_grad():

        output = model(
            input_tensor
        )

    print(
        f"Output shape: {output.shape}"
    )

    print(
        f"Output value: {output.item():.4f}"
    )

    # --------------------------------------------------
    # Verify expected output
    # --------------------------------------------------

    expected_shape = (1, 1)

    if tuple(output.shape) == expected_shape:

        print(
            "\nTransformer test PASSED."
        )

    else:

        print(
            "\nTransformer test FAILED."
        )


if __name__ == "__main__":
    main()

