import torch

from src.preprocessing.video_processor import VideoProcessor
from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.face_processor import FaceProcessor
from src.preprocessing.face_aligner import FaceAligner
from src.models.efficientnet_feature_extractor import (
    EfficientNetFeatureExtractor,
)


def main():
    video_path = "data/raw/test_video.mp4"
    output_path = "results/features.pt"

    print("=== Video Feature Extraction ===")

    # Initialize preprocessing components.
    detector = FaceDetector()

    face_processor = FaceProcessor(
        output_size=(224, 224),
        margin=0.2,
    )

    aligner = FaceAligner()

    # Initialize EfficientNet.
    extractor = EfficientNetFeatureExtractor()

    print("All models loaded successfully.")

    feature_sequence = []

    with VideoProcessor(video_path) as video:
        frames = video.sample_frames(
            num_frames=16,
        )

        print(f"Frames extracted: {len(frames)}")

        for i, frame in enumerate(frames):

            # 1. Detect faces.
            faces = detector.detect(frame)

            if not faces:
                print(f"Frame {i + 1:02d}: no face detected")
                continue

            # 2. Crop faces.
            cropped_faces = face_processor.process_frame(
                frame,
                faces,
            )

            if not cropped_faces:
                continue

            # 3. Align the first detected face.
            aligned_face = aligner.align(
                cropped_faces[0]
            )

            if aligned_face is None:
                print(f"Frame {i + 1:02d}: alignment failed")
                continue

            # 4. Convert BGR → RGB.
            rgb_face = aligned_face[:, :, ::-1].copy()

            # 5. Convert NumPy image to PyTorch tensor.
            face_tensor = torch.from_numpy(
                rgb_face
            ).permute(2, 0, 1)

            face_tensor = face_tensor.float() / 255.0

            # 6. Apply EfficientNet preprocessing.
            face_tensor = extractor.preprocess(
                face_tensor
            )

            # Add batch dimension.
            face_tensor = face_tensor.unsqueeze(0)

            # 7. Extract spatial features.
            features = extractor(face_tensor)

            # Remove batch dimension.
            features = features.squeeze(0)

            feature_sequence.append(features)

            print(
                f"Frame {i + 1:02d}: "
                f"feature shape = {features.shape}"
            )

    aligner.close()

    # Convert list to one tensor.
    feature_sequence = torch.stack(
        feature_sequence
    )

    print("\n=== Feature Extraction Summary ===")
    print(f"Feature sequence shape: {feature_sequence.shape}")

    # Save features.
    torch.save(
        feature_sequence,
        output_path,
    )

    print(f"Features saved to: {output_path}")


if __name__ == "__main__":
    main()