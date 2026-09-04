from src.training.video_dataset import VideoDataset


def main():
    print("=== Video Dataset Test ===")

    dataset = VideoDataset(
        root_dir="data/split/train",
        num_frames=16,
        image_size=(224, 224),
    )

    print(f"Dataset size: {len(dataset)}")

    video, label = dataset[0]

    print(f"Video tensor shape: {video.shape}")
    print(f"Label: {label.item()}")

    print("\n=== Test Summary ===")
    print("Expected video shape: torch.Size([16, 3, 224, 224])")
    print("Label: 0 = real, 1 = fake")

    dataset.close()


if __name__ == "__main__":
    main()