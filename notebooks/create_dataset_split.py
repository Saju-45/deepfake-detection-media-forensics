import os
import shutil

SOURCE_DIR = "data/raw"
OUTPUT_DIR = "data/split"

SPLITS = {
    "train": range(1, 38),
    "val": range(38, 46),
    "test": range(46, 54),
}


def copy_pair(split_name, pair_id):
    split_dir = os.path.join(OUTPUT_DIR, split_name)

    real_src = os.path.join(SOURCE_DIR, "real", f"v{pair_id}.mp4")
    fake_src = os.path.join(SOURCE_DIR, "fake", f"vs{pair_id}.mp4")

    real_dst = os.path.join(split_dir, "real", f"v{pair_id}.mp4")
    fake_dst = os.path.join(split_dir, "fake", f"vs{pair_id}.mp4")

    os.makedirs(os.path.dirname(real_dst), exist_ok=True)
    os.makedirs(os.path.dirname(fake_dst), exist_ok=True)

    shutil.copy2(real_src, real_dst)
    shutil.copy2(fake_src, fake_dst)


def main():
    print("=== Creating Dataset Split ===\n")

    if os.path.exists(OUTPUT_DIR):
        print(f"Removing existing split: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    for split_name, pair_ids in SPLITS.items():
        count = 0

        for pair_id in pair_ids:
            copy_pair(split_name, pair_id)
            count += 2

        print(f"{split_name.upper()}: {count} videos")

    print("\n=== Split Complete ===")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()