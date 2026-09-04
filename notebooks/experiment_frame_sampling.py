
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.training.video_dataset import VideoDataset
from src.models.efficientnet_feature_extractor import (
    EfficientNetFeatureExtractor,
)
from src.models.temporal_transformer import TemporalTransformer


# ==========================================================
# CONFIGURATION
# ==========================================================

TEST_DIR = "data/split/test"

FRAME_COUNTS = [8, 16, 32, 64]

BATCH_SIZE = 1

MODEL_PATH = "results/models/frame_sampling_transformer.pt"

LEARNING_RATE = 1e-4

EPOCHS = 5


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def extract_features(videos, spatial_model):
    """
    Extract EfficientNet-B0 spatial features.

    Input:
        videos: [B, T, 3, 224, 224]

    Output:
        features: [B, T, 1280]
    """

    batch_size = videos.shape[0]
    num_frames = videos.shape[1]

    frames = videos.view(
        batch_size * num_frames,
        3,
        224,
        224,
    )

    frames = spatial_model.preprocess(frames)

    features = spatial_model(frames)

    features = features.view(
        batch_size,
        num_frames,
        -1,
    )

    return features


# ==========================================================
# TRAIN MODEL FOR ONE FRAME COUNT
# ==========================================================

def train_model(
    train_loader,
    val_loader,
    spatial_model,
    num_frames,
):
    """
    Train a Temporal Transformer for a specific
    number of sampled frames.
    """

    print(
        f"\nTraining with {num_frames} frames..."
    )

    model = TemporalTransformer(
        input_size=1280,
        hidden_size=256,
        num_heads=8,
        num_layers=2,
        dropout=0.3,
        max_frames=num_frames,
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        train_labels = []
        train_predictions = []

        start_time = time.time()

        for videos, labels in train_loader:

            optimizer.zero_grad()

            features = extract_features(
                videos,
                spatial_model,
            )

            logits = model(features)

            loss = criterion(
                logits.squeeze(1),
                labels,
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            probabilities = torch.sigmoid(
                logits
            ).squeeze(1)

            predictions = (
                probabilities >= 0.5
            ).float()

            train_labels.extend(
                labels.tolist()
            )

            train_predictions.extend(
                predictions.tolist()
            )

        train_loss = (
            total_loss / len(train_loader)
        )

        train_accuracy = accuracy_score(
            train_labels,
            train_predictions,
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        model.eval()

        val_labels = []
        val_predictions = []

        val_loss_total = 0.0

        with torch.no_grad():

            for videos, labels in val_loader:

                features = extract_features(
                    videos,
                    spatial_model,
                )

                logits = model(features)

                loss = criterion(
                    logits.squeeze(1),
                    labels,
                )

                val_loss_total += loss.item()

                probabilities = torch.sigmoid(
                    logits
                ).squeeze(1)

                predictions = (
                    probabilities >= 0.5
                ).float()

                val_labels.extend(
                    labels.tolist()
                )

                val_predictions.extend(
                    predictions.tolist()
                )

        val_loss = (
            val_loss_total / len(val_loader)
        )

        val_accuracy = accuracy_score(
            val_labels,
            val_predictions,
        )

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                MODEL_PATH,
            )

    return model, best_val_accuracy


# ==========================================================
# TEST MODEL
# ==========================================================

def evaluate_test(
    model,
    test_loader,
    spatial_model,
):
    """
    Evaluate trained model on unseen test data.
    """

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for videos, labels in test_loader:

            features = extract_features(
                videos,
                spatial_model,
            )

            logits = model(features)

            probabilities = torch.sigmoid(
                logits
            ).squeeze(1)

            predictions = (
                probabilities >= 0.5
            ).float()

            all_labels.extend(
                labels.tolist()
            )

            all_predictions.extend(
                predictions.tolist()
            )

            all_probabilities.extend(
                probabilities.tolist()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print(
        "=== Frame Sampling Experiment ==="
    )

    print(
        f"Frame counts: {FRAME_COUNTS}"
    )

    print(
        f"Epochs per experiment: {EPOCHS}"
    )

    # ======================================================
    # LOAD EFFICIENTNET
    # ======================================================

    print(
        "\nLoading EfficientNet-B0..."
    )

    spatial_model = (
        EfficientNetFeatureExtractor()
    )

    spatial_model.eval()

    print(
        "EfficientNet loaded successfully."
    )

    # ======================================================
    # RESULTS
    # ======================================================

    results = []

    # ======================================================
    # RUN EACH FRAME COUNT
    # ======================================================

    for num_frames in FRAME_COUNTS:

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"FRAME COUNT: {num_frames}"
        )

        print(
            "=" * 60
        )

        # --------------------------------------------------
        # DATASETS
        # --------------------------------------------------

        train_dataset = VideoDataset(
            root_dir="data/split/train",
            num_frames=num_frames,
            image_size=(224, 224),
        )

        val_dataset = VideoDataset(
            root_dir="data/split/val",
            num_frames=num_frames,
            image_size=(224, 224),
        )

        test_dataset = VideoDataset(
            root_dir=TEST_DIR,
            num_frames=num_frames,
            image_size=(224, 224),
        )

        # --------------------------------------------------
        # DATA LOADERS
        # --------------------------------------------------

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

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        print(
            f"Train videos: {len(train_dataset)}"
        )

        print(
            f"Validation videos: {len(val_dataset)}"
        )

        print(
            f"Test videos: {len(test_dataset)}"
        )

        # --------------------------------------------------
        # TRAIN
        # --------------------------------------------------

        model, best_val_accuracy = train_model(
            train_loader,
            val_loader,
            spatial_model,
            num_frames,
        )

        # --------------------------------------------------
        # LOAD BEST MODEL
        # --------------------------------------------------

        if os.path.exists(MODEL_PATH):

            state_dict = torch.load(
                MODEL_PATH,
                map_location="cpu",
            )

            model.load_state_dict(
                state_dict
            )

        # --------------------------------------------------
        # TEST
        # --------------------------------------------------

        (
            test_accuracy,
            test_precision,
            test_recall,
            test_f1,
        ) = evaluate_test(
            model,
            test_loader,
            spatial_model,
        )

        # --------------------------------------------------
        # DISPLAY
        # --------------------------------------------------

        print(
            "\nResults:"
        )

        print(
            f"Best Validation Accuracy: "
            f"{best_val_accuracy * 100:.2f}%"
        )

        print(
            f"Test Accuracy: "
            f"{test_accuracy * 100:.2f}%"
        )

        print(
            f"Test Precision: "
            f"{test_precision * 100:.2f}%"
        )

        print(
            f"Test Recall: "
            f"{test_recall * 100:.2f}%"
        )

        print(
            f"Test F1: "
            f"{test_f1 * 100:.2f}%"
        )

        # --------------------------------------------------
        # SAVE RESULTS
        # --------------------------------------------------

        results.append(
            {
                "frames": num_frames,
                "val_accuracy": best_val_accuracy,
                "test_accuracy": test_accuracy,
                "precision": test_precision,
                "recall": test_recall,
                "f1": test_f1,
            }
        )

        # --------------------------------------------------
        # CLOSE DATASETS
        # --------------------------------------------------

        train_dataset.close()
        val_dataset.close()
        test_dataset.close()

    # ======================================================
    # FINAL COMPARISON
    # ======================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL FRAME SAMPLING RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"{'Frames':<10}"
        f"{'Val Acc':<12}"
        f"{'Test Acc':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
    )

    print(
        "-" * 70
    )

    for result in results:

        print(
            f"{result['frames']:<10}"
            f"{result['val_accuracy'] * 100:<12.2f}"
            f"{result['test_accuracy'] * 100:<12.2f}"
            f"{result['precision'] * 100:<12.2f}"
            f"{result['recall'] * 100:<12.2f}"
            f"{result['f1'] * 100:<12.2f}"
        )

    # ======================================================
    # BEST FRAME COUNT
    # ======================================================

    best_result = max(
        results,
        key=lambda x: x["test_accuracy"],
    )

    print(
        "\nBest frame count based on "
        "test accuracy:"
    )

    print(
        f"{best_result['frames']} frames"
    )

    print(
        f"Test Accuracy: "
        f"{best_result['test_accuracy'] * 100:.2f}%"
    )

    print(
        "\n=== Experiment Complete ==="
    )


# ==========================================================
# PROGRAM ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
