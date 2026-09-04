from src.preprocessing.video_processor import VideoProcessor
from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.face_processor import FaceProcessor
from src.preprocessing.face_aligner import FaceAligner


def main():
    video_path = "data/split/train/fake/vs33.mp4"

    print("=== Face Processing Diagnostic ===")
    print(f"Video: {video_path}\n")

    detector = FaceDetector()
    processor = FaceProcessor(
        output_size=(224, 224),
        margin=0.2,
    )
    aligner = FaceAligner()

    success_count = 0

    with VideoProcessor(video_path) as video:
        frames = video.sample_frames(num_frames=16)

        for i, frame in enumerate(frames):
            faces = detector.detect(frame)

            if not faces:
                print(f"Frame {i + 1:02d}: detection FAILED")
                continue

            # Use largest face
            faces = sorted(
                faces,
                key=lambda box: box[2] * box[3],
                reverse=True,
            )

            cropped = processor.process_frame(
                frame,
                [faces[0]],
            )

            if not cropped:
                print(f"Frame {i + 1:02d}: crop FAILED")
                continue

            aligned = aligner.align(cropped[0])

            if aligned is None:
                print(f"Frame {i + 1:02d}: alignment FAILED")
                continue

            success_count += 1

            print(
                f"Frame {i + 1:02d}: "
                f"OK → detected → cropped → aligned"
            )

    aligner.close()

    print("\n=== Summary ===")
    print(f"Usable frames: {success_count}/16")


if __name__ == "__main__":
    main()