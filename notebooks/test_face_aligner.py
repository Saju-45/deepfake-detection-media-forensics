from src.preprocessing.video_processor import VideoProcessor
from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.face_processor import FaceProcessor
from src.preprocessing.face_aligner import FaceAligner


def main():
    video_path = "data/raw/test_video.mp4"

    detector = FaceDetector()
    processor = FaceProcessor(
        output_size=(224, 224),
        margin=0.2,
    )
    aligner = FaceAligner()

    with VideoProcessor(video_path) as video:
        frames = video.sample_frames(num_frames=16)

        print("=== Face Alignment Test ===")

        total_detected = 0
        total_cropped = 0
        total_aligned = 0

        for i, frame in enumerate(frames):

            # Step 1: Detect face
            faces = detector.detect(frame)
            total_detected += len(faces)

            # Step 2: Crop and resize faces
            cropped_faces = processor.process_frame(
                frame,
                faces,
            )
            total_cropped += len(cropped_faces)

            # Step 3: Align faces
            aligned_faces = []

            for face in cropped_faces:
                aligned_face = aligner.align(face)

                if aligned_face is not None:
                    aligned_faces.append(aligned_face)

            total_aligned += len(aligned_faces)

            print(
                f"Frame {i + 1:02d}: "
                f"{len(faces)} detected → "
                f"{len(cropped_faces)} cropped → "
                f"{len(aligned_faces)} aligned"
            )

            if aligned_faces:
                print(
                    f"           Aligned face shape: "
                    f"{aligned_faces[0].shape}"
                )

        print("\n=== Alignment Summary ===")
        print(f"Frames processed: {len(frames)}")
        print(f"Faces detected: {total_detected}")
        print(f"Faces cropped: {total_cropped}")
        print(f"Faces aligned: {total_aligned}")

    aligner.close()


if __name__ == "__main__":
    main()