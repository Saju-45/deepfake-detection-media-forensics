
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
from src.models.temporal_lstm import TemporalLSTM


TEST_DIR = "data/split/test"
MODEL_PATH = "results/models/best_lstm.pt"

NUM_FRAMES = 16
BATCH_SIZE = 1


def main():

    print("=== Testing Deepfake Detection Model ===")

    # ==================================================
    # 1. LOAD TEST DATASET
    # ==================================================

    print("\nLoading test dataset...")

    test_dataset = VideoDataset(
        root_dir=TEST_DIR,
        num_frames=NUM_FRAMES,
        image_size=(224, 224),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(f"Test videos: {len(test_dataset)}")

    # ==================================================
    # 2. LOAD EFFICIENTNET
    # ==================================================

    print("\nLoading EfficientNet-B0...")

    feature_extractor = EfficientNetFeatureExtractor()
    feature_extractor.eval()

    print("EfficientNet loaded successfully.")

    # ==================================================
    # 3. LOAD TRAINED LSTM
    # ==================================================

    print("\nLoading trained LSTM...")

    temporal_model = TemporalLSTM()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    temporal_model.load_state_dict(checkpoint)
    temporal_model.eval()

    print("Trained LSTM loaded successfully.")

    # ==================================================
    # 4. EVALUATE TEST DATA
    # ==================================================

    all_predictions = []
    all_labels = []

    print("\n=== Test Predictions ===")

    with torch.no_grad():

        for batch_index, (videos, labels) in enumerate(test_loader):

            # videos:
            # [1, 16, 3, 224, 224]

            batch_size = videos.shape[0]
            num_frames = videos.shape[1]

            # --------------------------------------------------
            # Combine batch + frames
            #
            # [1, 16, 3, 224, 224]
            #       ↓
            # [16, 3, 224, 224]
            # --------------------------------------------------

            frames = videos.view(
                batch_size * num_frames,
                3,
                224,
                224,
            )

            # --------------------------------------------------
            # Apply EfficientNet ImageNet preprocessing
            # --------------------------------------------------

            frames = feature_extractor.preprocess(frames)

            # --------------------------------------------------
            # Extract spatial features
            #
            # [16, 3, 224, 224]
            #       ↓
            # [16, 1280]
            # --------------------------------------------------

            spatial_features = feature_extractor(frames)

            # --------------------------------------------------
            # Restore temporal dimension
            #
            # [16, 1280]
            #       ↓
            # [1, 16, 1280]
            # --------------------------------------------------

            temporal_features = spatial_features.view(
                batch_size,
                num_frames,
                -1,
            )

            # --------------------------------------------------
            # LSTM prediction
            # --------------------------------------------------

            logits = temporal_model(temporal_features)

            probability = torch.sigmoid(logits).item()

            prediction = 1 if probability >= 0.5 else 0

            actual_label = int(labels.item())

            # Store predictions and labels
            all_predictions.append(prediction)
            all_labels.append(actual_label)

            predicted_name = (
                "FAKE" if prediction == 1 else "REAL"
            )

            actual_name = (
                "FAKE" if actual_label == 1 else "REAL"
            )

            print(
                f"Video {batch_index + 1:02d}: "
                f"Prediction={predicted_name} "
                f"Probability={probability:.4f} "
                f"Actual={actual_name}"
            )

    # ==================================================
    # 5. CALCULATE METRICS
    # ==================================================

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

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
    )

    # ==================================================
    # 6. DISPLAY RESULTS
    # ==================================================

    print("\n=== Test Results ===")

    print(f"Test Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1 Score       : {f1:.4f}")

    print("\n=== Confusion Matrix ===")

    print("                 Predicted")
    print("                 REAL  FAKE")
    print(
        f"Actual REAL     {matrix[0][0]:4d}  {matrix[0][1]:4d}"
    )
    print(
        f"Actual FAKE     {matrix[1][0]:4d}  {matrix[1][1]:4d}"
    )

    print("\n=== Testing Complete ===")

    # ==================================================
    # 7. CLOSE DATASET
    # ==================================================

    test_dataset.close()


if __name__ == "__main__":
    main()


