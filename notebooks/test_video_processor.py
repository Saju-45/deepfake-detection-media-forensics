from src.preprocessing.video_processor import VideoProcessor


def main():
    video_path = "data/raw/test_video.mp4"

    with VideoProcessor(video_path) as processor:
        metadata = processor.get_metadata()

        print("=== Video Metadata ===")
        for key, value in metadata.items():
            print(f"{key}: {value}")

        frames = processor.sample_frames(
            num_frames=16,
            resize=(224, 224),
        )

        print("\n=== Frame Extraction ===")
        print(f"Frames requested: 16")
        print(f"Frames extracted: {len(frames)}")

        if frames:
            print(f"First frame shape: {frames[0].shape}")


if __name__ == "__main__":
    main()