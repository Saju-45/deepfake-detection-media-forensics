from src.preprocessing.video_processor import VideoProcessor
from src.preprocessing.face_detector import FaceDetector


def main():
    video_path = "data/raw/test_video.mp4"

    detector = FaceDetector()

    with VideoProcessor(video_path) as processor:
        frames = processor.sample_frames(
        num_frames=16,
)

        print("=== Face Detection ===")

        total_faces = 0

        for i, frame in enumerate(frames):
            faces = detector.detect(frame)

            print(f"Frame {i + 1:02d}: {len(faces)} face(s)")

            total_faces += len(faces)

        print("\n=== Detection Summary ===")
        print(f"Frames processed: {len(frames)}")
        print(f"Total faces detected: {total_faces}")


if __name__ == "__main__":
    main()