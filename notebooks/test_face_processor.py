from src.preprocessing.video_processor import VideoProcessor
from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.face_processor import FaceProcessor


def main():
    video_path = "data/raw/test_video.mp4"

    detector = FaceDetector()
    processor = FaceProcessor(
        output_size=(224, 224),
        margin=0.2,
    )

    with VideoProcessor(video_path) as video:
        frames = video.sample_frames(num_frames=16)

        print("=== Face Cropping Test ===")

        total_faces = 0
        total_processed = 0

        for i, frame in enumerate(frames):
            faces = detector.detect(frame)

            processed_faces = processor.process_frame(
                frame,
                faces,
            )

            print(
                f"Frame {i + 1:02d}: "
                f"{len(faces)} detected → "
                f"{len(processed_faces)} cropped"
            )

            total_faces += len(faces)
            total_processed += len(processed_faces)

            if processed_faces:
                print(
                    f"           First face shape: "
                    f"{processed_faces[0].shape}"
                )

        print("\n=== Cropping Summary ===")
        print(f"Frames processed: {len(frames)}")
        print(f"Faces detected: {total_faces}")
        print(f"Faces cropped: {total_processed}")


if __name__ == "__main__":
    main()