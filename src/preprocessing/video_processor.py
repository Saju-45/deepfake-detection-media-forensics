from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


class VideoProcessor:
    """
    Handles video loading and frame sampling.

    The processor does not perform deepfake detection itself.
    Its job is to reliably convert a video into usable frames.
    """

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {self.video_path}"
            )

        self.capture = cv2.VideoCapture(str(self.video_path))

        if not self.capture.isOpened():
            raise ValueError(
                f"Unable to open video: {self.video_path}"
            )

    def get_metadata(self) -> dict:
        """Return basic video metadata."""

        fps = self.capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(
            self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        width = int(
            self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        height = int(
            self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        duration = frame_count / fps if fps > 0 else 0

        return {
            "path": str(self.video_path),
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_seconds": duration,
        }

    def sample_frames(
        self,
        num_frames: int = 16,
        resize: Tuple[int, int] | None = None,
    ) -> List[np.ndarray]:
        """
        Uniformly sample frames from the video.

        Args:
            num_frames: Number of frames to extract.
            resize: Optional (width, height).

        Returns:
            List of OpenCV BGR frames.
        """

        if num_frames <= 0:
            raise ValueError("num_frames must be greater than 0")

        metadata = self.get_metadata()
        total_frames = metadata["frame_count"]

        if total_frames == 0:
            raise ValueError("Video contains no frames")

        frame_indices = np.linspace(
            0,
            total_frames - 1,
            min(num_frames, total_frames),
            dtype=int,
        )

        frames = []

        for index in frame_indices:
            self.capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(index),
            )

            success, frame = self.capture.read()

            if not success:
                continue

            if resize is not None:
                frame = cv2.resize(
                    frame,
                    resize,
                    interpolation=cv2.INTER_AREA,
                )

            frames.append(frame)

        return frames

    def close(self):
        """Release the video resource."""
        if self.capture is not None:
            self.capture.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()