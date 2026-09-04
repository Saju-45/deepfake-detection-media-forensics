import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.video_dataset import VideoDataset
from src.models.efficientnet_feature_extractor import (
    EfficientNetFeatureExtractor,
)
from src.models.temporal_lstm import TemporalLSTM


NUM_FRAMES = 16
BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 1e-4

TRAIN_DIR = "data/split/train"
VAL_DIR = "data/split/val"

MODEL_DIR = "results/models"

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_lstm.pt",
)


def extract_features(video_batch, spatial_model):
    """
    Convert video frames into EfficientNet features.

    Input:
        [B, T, 3, 224, 224]

    Output:
        [B, T, 1280]
    """

    batch_size, num_frames, channels, height, width = (
        video_batch.shape
    )

    # --------------------------------------------------
    # Combine batch and temporal dimensions
    #
    # [B, T, 3, 224, 224]
    #        ↓
    # [B*T, 3, 224, 224]
    # --------------------------------------------------

    frames = video_batch.view(
        batch_size * num_frames,
        channels,
        height,
        width,
    )

    # --------------------------------------------------
    # EfficientNet preprocessing
    #
    # VideoDataset produces pixel values in [0, 1].
    #
    # EfficientNet-B0 pretrained weights expect
    # ImageNet-normalized input.
    # --------------------------------------------------

    frames = spatial_model.preprocess(
        frames
    )

    # --------------------------------------------------
    # Extract spatial features
    #
    # [B*T, 3, 224, 224]
    #        ↓
    # [B*T, 1280]
    # --------------------------------------------------

    with torch.no_grad():

        features = spatial_model(
            frames
        )

    # --------------------------------------------------
    # Restore temporal dimension
    #
    # [B*T, 1280]
    #        ↓
    # [B, T, 1280]
    # --------------------------------------------------

    features = features.view(
        batch_size,
        num_frames,
        -1,
    )

    return features


def evaluate(
    dataset,
    spatial_model,
    temporal_model,
    criterion,
):
    """
    Evaluate the temporal model on a dataset.
    """

    temporal_model.eval()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for videos, labels in loader:

            # --------------------------------------------------
            # Extract EfficientNet spatial features
            # --------------------------------------------------

            features = extract_features(
                videos,
                spatial_model,
            )

            # --------------------------------------------------
            # Temporal LSTM prediction
            # --------------------------------------------------

            logits = temporal_model(
                features
            ).squeeze(1)

            # --------------------------------------------------
            # Calculate loss
            # --------------------------------------------------

            loss = criterion(
                logits,
                labels,
            )

            total_loss += loss.item()

            # --------------------------------------------------
            # Convert logits to probabilities
            # --------------------------------------------------

            probabilities = torch.sigmoid(
                logits
            )

            # --------------------------------------------------
            # Convert probabilities to predictions
            #
            # >= 0.5 → fake
            # < 0.5  → real
            # --------------------------------------------------

            predictions = (
                probabilities >= 0.5
            ).float()

            # --------------------------------------------------
            # Accuracy
            # --------------------------------------------------

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = (
        total_loss / len(loader)
    )

    accuracy = correct / total

    return average_loss, accuracy


def main():

    print("=== LSTM Training ===\n")

    # ==================================================
    # DEVICE
    # ==================================================

    device = torch.device("cpu")

    print(
        f"Device: {device}"
    )

    # ==================================================
    # CREATE MODEL DIRECTORY
    # ==================================================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    # ==================================================
    # DATASETS
    # ==================================================

    print("\nLoading datasets...")

    train_dataset = VideoDataset(
        root_dir=TRAIN_DIR,
        num_frames=NUM_FRAMES,
        image_size=(224, 224),
    )

    val_dataset = VideoDataset(
        root_dir=VAL_DIR,
        num_frames=NUM_FRAMES,
        image_size=(224, 224),
    )

    print(
        f"Training videos: {len(train_dataset)}"
    )

    print(
        f"Validation videos: {len(val_dataset)}"
    )

    # ==================================================
    # DATA LOADER
    # ==================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # ==================================================
    # SPATIAL MODEL
    # ==================================================

    print(
        "\nLoading EfficientNet-B0..."
    )

    spatial_model = EfficientNetFeatureExtractor()

    spatial_model.to(device)

    spatial_model.eval()

    print(
        "EfficientNet-B0: FROZEN"
    )

    # ==================================================
    # TEMPORAL MODEL
    # ==================================================

    print(
        "Loading Temporal LSTM..."
    )

    temporal_model = TemporalLSTM(
        input_size=1280,
        hidden_size=256,
        num_layers=2,
        dropout=0.3,
    )

    temporal_model.to(device)

    print(
        "Temporal LSTM: TRAINABLE"
    )

    # ==================================================
    # LOSS FUNCTION
    # ==================================================

    criterion = nn.BCEWithLogitsLoss()

    # ==================================================
    # OPTIMIZER
    # ==================================================

    optimizer = torch.optim.Adam(
        temporal_model.parameters(),
        lr=LEARNING_RATE,
    )

    # ==================================================
    # BEST VALIDATION ACCURACY
    # ==================================================

    best_val_accuracy = 0.0

    # ==================================================
    # TRAINING LOOP
    # ==================================================

    for epoch in range(EPOCHS):

        temporal_model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        # --------------------------------------------------
        # Training batches
        # --------------------------------------------------

        for batch_index, (
            videos,
            labels,
        ) in enumerate(
            train_loader,
            start=1,
        ):

            videos = videos.to(device)

            labels = labels.to(device)

            # --------------------------------------------------
            # Extract spatial features
            # --------------------------------------------------

            features = extract_features(
                videos,
                spatial_model,
            )

            # --------------------------------------------------
            # Temporal prediction
            # --------------------------------------------------

            logits = temporal_model(
                features
            ).squeeze(1)

            # --------------------------------------------------
            # Loss
            # --------------------------------------------------

            loss = criterion(
                logits,
                labels,
            )

            # --------------------------------------------------
            # Backpropagation
            # --------------------------------------------------

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # --------------------------------------------------
            # Statistics
            # --------------------------------------------------

            running_loss += loss.item()

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= 0.5
            ).float()

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            # --------------------------------------------------
            # Progress
            # --------------------------------------------------

            if batch_index % 10 == 0:

                print(
                    f"  Batch "
                    f"{batch_index}/"
                    f"{len(train_loader)}"
                )

        # ==================================================
        # TRAINING METRICS
        # ==================================================

        train_loss = (
            running_loss
            / len(train_loader)
        )

        train_accuracy = (
            correct / total
        )

        # ==================================================
        # VALIDATION
        # ==================================================

        val_loss, val_accuracy = evaluate(
            val_dataset,
            spatial_model,
            temporal_model,
            criterion,
        )

        # ==================================================
        # DISPLAY METRICS
        # ==================================================

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Val Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy: "
            f"{val_accuracy:.4f}"
        )

        # ==================================================
        # SAVE BEST MODEL
        # ==================================================

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = (
                val_accuracy
            )

            torch.save(
                temporal_model.state_dict(),
                BEST_MODEL_PATH,
            )

            print(
                f"✓ Best model saved: "
                f"{BEST_MODEL_PATH}"
            )

    # ==================================================
    # TRAINING COMPLETE
    # ==================================================

    print(
        "\n=== Training Complete ==="
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Model: "
        f"{BEST_MODEL_PATH}"
    )

    # ==================================================
    # RELEASE RESOURCES
    # ==================================================

    train_dataset.close()

    val_dataset.close()


if __name__ == "__main__":
    main()