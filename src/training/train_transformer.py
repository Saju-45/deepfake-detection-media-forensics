
print("TRANSFORMER TRAINING FILE LOADED")

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.video_dataset import VideoDataset
from src.models.efficientnet_feature_extractor import EfficientNetFeatureExtractor
from src.models.temporal_transformer import TemporalTransformer


# ==========================================================
# CONFIGURATION
# ==========================================================

TRAIN_DIR = "data/split/train"
VAL_DIR = "data/split/val"

MODEL_PATH = "results/models/best_transformer.pt"

NUM_FRAMES = 16
BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 1e-4


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def extract_features(videos, spatial_model):
    """
    Extract EfficientNet-B0 spatial features from videos.

    Input:
        videos: [batch, frames, 3, 224, 224]

    Output:
        features: [batch, frames, 1280]
    """

    batch_size = videos.shape[0]
    num_frames = videos.shape[1]

    # ------------------------------------------------------
    # Combine batch and frame dimensions
    #
    # [B, T, 3, 224, 224]
    #        ↓
    # [B*T, 3, 224, 224]
    # ------------------------------------------------------

    frames = videos.view(
        batch_size * num_frames,
        3,
        224,
        224,
    )

    # ------------------------------------------------------
    # EfficientNet ImageNet preprocessing
    # ------------------------------------------------------

    frames = spatial_model.preprocess(frames)

    # ------------------------------------------------------
    # Extract spatial features
    #
    # [B*T, 3, 224, 224]
    #        ↓
    # [B*T, 1280]
    # ------------------------------------------------------

    features = spatial_model(frames)

    # ------------------------------------------------------
    # Restore temporal dimension
    #
    # [B*T, 1280]
    #        ↓
    # [B, T, 1280]
    # ------------------------------------------------------

    features = features.view(
        batch_size,
        num_frames,
        -1,
    )

    return features


# ==========================================================
# VALIDATION
# ==========================================================

def evaluate(
    temporal_model,
    spatial_model,
    data_loader,
    criterion,
):
    """
    Evaluate Transformer on validation data.
    """

    temporal_model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for videos, labels in data_loader:

            # --------------------------------------------------
            # Extract EfficientNet features
            # --------------------------------------------------

            features = extract_features(
                videos,
                spatial_model,
            )

            # --------------------------------------------------
            # Transformer prediction
            # --------------------------------------------------

            logits = temporal_model(
                features
            )

            # --------------------------------------------------
            # Calculate loss
            # --------------------------------------------------

            loss = criterion(
                logits.squeeze(1),
                labels,
            )

            total_loss += loss.item()

            # --------------------------------------------------
            # Convert logits to predictions
            # --------------------------------------------------

            probabilities = torch.sigmoid(
                logits
            ).squeeze(1)

            predictions = (
                probabilities >= 0.5
            ).float()

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = total_loss / len(data_loader)

    accuracy = correct / total

    return average_loss, accuracy


# ==========================================================
# MAIN TRAINING FUNCTION
# ==========================================================

def main():

    print("=== Temporal Transformer Training ===")

    # ======================================================
    # 1. LOAD TRAINING DATASET
    # ======================================================

    print("\nLoading training dataset...")

    train_dataset = VideoDataset(
        root_dir=TRAIN_DIR,
        num_frames=NUM_FRAMES,
        image_size=(224, 224),
    )

    print(
        f"Training videos: {len(train_dataset)}"
    )

    # ======================================================
    # 2. LOAD VALIDATION DATASET
    # ======================================================

    print("\nLoading validation dataset...")

    val_dataset = VideoDataset(
        root_dir=VAL_DIR,
        num_frames=NUM_FRAMES,
        image_size=(224, 224),
    )

    print(
        f"Validation videos: {len(val_dataset)}"
    )

    # ======================================================
    # 3. CREATE DATA LOADERS
    # ======================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # ======================================================
    # 4. LOAD EFFICIENTNET
    # ======================================================

    print("\nLoading EfficientNet-B0...")

    spatial_model = EfficientNetFeatureExtractor()

    spatial_model.eval()

    print(
        "EfficientNet loaded successfully."
    )

    # ======================================================
    # 5. LOAD TEMPORAL TRANSFORMER
    # ======================================================

    print("\nLoading Temporal Transformer...")

    temporal_model = TemporalTransformer()

    temporal_model.train()

    print(
        "Temporal Transformer loaded successfully."
    )

    # ======================================================
    # 6. LOSS FUNCTION
    # ======================================================

    criterion = nn.BCEWithLogitsLoss()

    # ======================================================
    # 7. OPTIMIZER
    # ======================================================

    optimizer = torch.optim.Adam(
        temporal_model.parameters(),
        lr=LEARNING_RATE,
    )

    # ======================================================
    # 8. BEST MODEL TRACKING
    # ======================================================

    best_val_accuracy = 0.0

    model_directory = os.path.dirname(
        MODEL_PATH
    )

    os.makedirs(
        model_directory,
        exist_ok=True,
    )

    # ======================================================
    # 9. TRAINING LOOP
    # ======================================================

    for epoch in range(EPOCHS):

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        temporal_model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        # --------------------------------------------------
        # TRAINING
        # --------------------------------------------------

        for batch_index, (videos, labels) in enumerate(
            train_loader
        ):

            optimizer.zero_grad()

            # ------------------------------------------------
            # Extract EfficientNet spatial features
            # ------------------------------------------------

            features = extract_features(
                videos,
                spatial_model,
            )

            # ------------------------------------------------
            # Transformer
            # ------------------------------------------------

            logits = temporal_model(
                features
            )

            # ------------------------------------------------
            # Calculate loss
            # ------------------------------------------------

            loss = criterion(
                logits.squeeze(1),
                labels,
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            loss.backward()

            optimizer.step()

            # ------------------------------------------------
            # Training statistics
            # ------------------------------------------------

            total_loss += loss.item()

            probabilities = torch.sigmoid(
                logits
            ).squeeze(1)

            predictions = (
                probabilities >= 0.5
            ).float()

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (batch_index + 1) % 10 == 0:

                print(
                    f"  Processed "
                    f"{batch_index + 1}/"
                    f"{len(train_loader)} videos"
                )

        # --------------------------------------------------
        # Training metrics
        # --------------------------------------------------

        train_loss = (
            total_loss / len(train_loader)
        )

        train_accuracy = (
            correct / total
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        val_loss, val_accuracy = evaluate(
            temporal_model,
            spatial_model,
            val_loader,
            criterion,
        )

        # --------------------------------------------------
        # Display metrics
        # --------------------------------------------------

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f}"
        )

        print(
            f"Val Accuracy: "
            f"{val_accuracy:.4f}"
        )

        # --------------------------------------------------
        # Save best model
        # --------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                temporal_model.state_dict(),
                MODEL_PATH,
            )

            print(
                "Best Transformer model saved."
            )

    # ======================================================
    # 10. TRAINING COMPLETE
    # ======================================================

    print(
        "\n=== Training Complete ==="
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    # ======================================================
    # 11. CLOSE DATASET RESOURCES
    # ======================================================

    train_dataset.close()
    val_dataset.close()


# ==========================================================
# PROGRAM ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()