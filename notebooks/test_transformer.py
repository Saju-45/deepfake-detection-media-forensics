
import os

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.training.video_dataset import VideoDataset
from src.models.efficientnet_feature_extractor import EfficientNetFeatureExtractor
from src.models.temporal_transformer import TemporalTransformer


# ==========================================================
# CONFIGURATION
# ==========================================================

TEST_DIR = "data/split/test"
MODEL_PATH = "results/models/best_transformer.pt"

NUM_FRAMES = 16
BATCH_SIZE = 1


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
# MAIN
# ==========================================================

def main():

    print("=== Temporal Transformer Test ===")

    # ======================================================
    # 1. CHECK MODEL FILE
    # ======================================================

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Transformer model not found: {MODEL_PATH}"
        )

    print(
        f"Model found: {MODEL_PATH}"
    )

    # ======================================================
    # 2. LOAD TEST DATASET
    # ======================================================

    print("\nLoading test dataset...")

    test_dataset = VideoDataset(
        root_dir=TEST_DIR,
        num_frames=NUM_FRAMES,
        image_size=(224, 224),
    )

    print(
        f"Test videos: {len(test_dataset)}"
    )

    # ======================================================
    # 3. CREATE DATA LOADER
    # ======================================================

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # ======================================================
    # 4. LOAD EFFICIENTNET
    # ======================================================

    print("\nLoading EfficientNet-B0...")

    feature_extractor = (
        EfficientNetFeatureExtractor()
    )

    feature_extractor.eval()

    print(
        "EfficientNet loaded successfully."
    )

    # ======================================================
    # 5. LOAD TRANSFORMER
    # ======================================================

    print("\nLoading Temporal Transformer...")

    model = TemporalTransformer()

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    model.load_state_dict(state_dict)

    model.eval()

    print(
        "Transformer model loaded successfully."
    )

    # ======================================================
    # 6. EVALUATION
    # ======================================================

    all_labels = []
    all_predictions = []
    all_probabilities = []

    print("\n=== Predictions ===")

    with torch.no_grad():

        for index, (videos, labels) in enumerate(
            test_loader
        ):

            # ------------------------------------------------
            # Extract EfficientNet features
            # ------------------------------------------------

            features = extract_features(
                videos,
                feature_extractor,
            )

            # ------------------------------------------------
            # Transformer prediction
            # ------------------------------------------------

            logits = model(features)

            probability = torch.sigmoid(
                logits
            ).item()

            # ------------------------------------------------
            # Classification threshold
            # ------------------------------------------------

            prediction = (
                1 if probability >= 0.5 else 0
            )

            actual = int(labels.item())

            # ------------------------------------------------
            # Store results
            # ------------------------------------------------

            all_labels.append(actual)
            all_predictions.append(prediction)
            all_probabilities.append(
                probability
            )

            # ------------------------------------------------
            # Convert labels to names
            # ------------------------------------------------

            actual_name = (
                "FAKE" if actual == 1 else "REAL"
            )

            prediction_name = (
                "FAKE"
                if prediction == 1
                else "REAL"
            )

            filename = os.path.basename(
                test_dataset.video_paths[index]
            )

            print(
                f"{index + 1:02d}. "
                f"{filename:<20} "
                f"Actual: {actual_name:<5} "
                f"Predicted: {prediction_name:<5} "
                f"Fake Probability: {probability:.4f}"
            )

    # ======================================================
    # 7. CALCULATE METRICS
    # ======================================================

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

    # ======================================================
    # 8. CONFUSION MATRIX
    # ======================================================

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1],
    )

    # ======================================================
    # 9. PRINT RESULTS
    # ======================================================

    print("\n=== Test Results ===")

    print(
        f"Test Accuracy : "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision      : "
        f"{precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall         : "
        f"{recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1 Score       : "
        f"{f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    # ======================================================
    # 10. CONFUSION MATRIX
    # ======================================================

    print("\n=== Confusion Matrix ===")

    print(
        "                 Predicted"
    )

    print(
        "                 REAL  FAKE"
    )

    print(
        f"Actual REAL       "
        f"{cm[0][0]:4d}  "
        f"{cm[0][1]:4d}"
    )

    print(
        f"Actual FAKE       "
        f"{cm[1][0]:4d}  "
        f"{cm[1][1]:4d}"
    )

    # ======================================================
    # 11. INTERPRET CONFUSION MATRIX
    # ======================================================

    true_negative = cm[0][0]
    false_positive = cm[0][1]
    false_negative = cm[1][0]
    true_positive = cm[1][1]

    print("\n=== Classification Breakdown ===")

    print(
        f"True Negatives  (REAL → REAL): "
        f"{true_negative}"
    )

    print(
        f"False Positives (REAL → FAKE): "
        f"{false_positive}"
    )

    print(
        f"False Negatives (FAKE → REAL): "
        f"{false_negative}"
    )

    print(
        f"True Positives  (FAKE → FAKE): "
        f"{true_positive}"
    )

    # ======================================================
    # 12. CLOSE DATASET RESOURCES
    # ======================================================

    test_dataset.close()

    print("\n=== Test Complete ===")


# ==========================================================
# PROGRAM ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()

