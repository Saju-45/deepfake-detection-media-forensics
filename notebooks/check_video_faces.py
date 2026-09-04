from src.preprocessing.video_processor import VideoProcessor
from src.preprocessing.face_detector import FaceDetector


def main():
    video_path = "data/split/train/fake/vs33.mp4"

    print("=== Face Detection Diagnostic ===")
    print(f"Video: {video_path}\n")

    detector = FaceDetector()

    with VideoProcessor(video_path) as processor:
        metadata = processor.get_metadata()

        print("Video Metadata:")
        for key, value in metadata.items():
            print(f"{key}: {value}")

        frames = processor.sample_frames(num_frames=16)

        print("\nFace Detection:")

        total_faces = 0

        for i, frame in enumerate(frames):
            faces = detector.detect(frame)

            print(
                f"Frame {i + 1:02d}: "
                f"{len(faces)} face(s)"
            )

            total_faces += len(faces)

        print("\n=== Summary ===")
        print(f"Frames checked: {len(frames)}")
        print(f"Total faces detected: {total_faces}")


if __name__ == "__main__":
    main()