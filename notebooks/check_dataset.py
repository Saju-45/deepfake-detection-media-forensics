import os
import cv2


DATASET_DIR = "data/raw"


def inspect_video(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": fps,
        "frames": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
    }


def main():
    print("=== Dataset Inspection ===\n")

    total = 0
    real_count = 0
    fake_count = 0
    failed = 0

    for label in ["real", "fake"]:

        folder = os.path.join(DATASET_DIR, label)

        print(f"--- {label.upper()} ---")

        for filename in sorted(os.listdir(folder)):

            if not filename.lower().endswith(".mp4"):
                continue

            path = os.path.join(folder, filename)

            info = inspect_video(path)

            if info is None:
                print(f"ERROR: {filename}")
                failed += 1
                continue

            total += 1

            if label == "real":
                real_count += 1
            else:
                fake_count += 1

            print(
                f"{filename}: "
                f"{info['frames']} frames, "
                f"{info['fps']:.1f} FPS, "
                f"{info['width']}x{info['height']}, "
                f"{info['duration']:.2f}s"
            )

    print("\n=== Dataset Summary ===")
    print(f"Total videos: {total}")
    print(f"Real videos: {real_count}")
    print(f"Fake videos: {fake_count}")
    print(f"Failed videos: {failed}")


if __name__ == "__main__":
    main()