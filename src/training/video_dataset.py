import os

import cv2
import torch
from torch.utils.data import Dataset

from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.face_processor import FaceProcessor
from src.preprocessing.face_aligner import FaceAligner


class VideoDataset(Dataset):
    """
    PyTorch dataset for real/fake video classification.

    Each item:
        video: Tensor of shape [num_frames, 3, 224, 224]
        label: 0 = real, 1 = fake
    """

    def __init__(
        self,
        root_dir,
        num_frames=16,
        image_size=(224, 224),
    ):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.image_size = image_size

        self.video_paths = []
        self.labels = []

        self._collect_videos()

        # Face detector
        self.face_detector = FaceDetector()

        # Face crop processor
        self.face_processor = FaceProcessor(
            output_size=image_size,
            margin=0.2,
        )

        # Face landmark / alignment model
        self.face_aligner = FaceAligner()

    def _collect_videos(self):
        """
        Collect video paths and labels.

        real = 0
        fake = 1
        """

        class_map = {
            "real": 0,
            "fake": 1,
        }

        for class_name, label in class_map.items():

            class_dir = os.path.join(
                self.root_dir,
                class_name,
            )

            if not os.path.isdir(class_dir):
                raise FileNotFoundError(
                    f"Dataset folder not found: {class_dir}"
                )

            for filename in sorted(
                os.listdir(class_dir)
            ):

                if filename.lower().endswith(".mp4"):

                    video_path = os.path.join(
                        class_dir,
                        filename,
                    )

                    self.video_paths.append(
                        video_path
                    )

                    self.labels.append(label)

        if not self.video_paths:
            raise RuntimeError(
                f"No MP4 videos found in {self.root_dir}"
            )

    def __len__(self):
        """
        Return the number of videos in the dataset.
        """

        return len(self.video_paths)

    def _sample_frames(self, video_path):
        """
        Sample evenly spaced frames from a video.

        Returns:
            list of OpenCV BGR frames
        """

        capture = cv2.VideoCapture(
            video_path
        )

        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open video: {video_path}"
            )

        frame_count = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if frame_count <= 0:
            capture.release()

            raise RuntimeError(
                f"Video contains no frames: {video_path}"
            )

        # Evenly distribute frame indices
        indices = torch.linspace(
            0,
            frame_count - 1,
            steps=self.num_frames,
        ).long().tolist()

        frames = []

        for index in indices:

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(index),
            )

            success, frame = capture.read()

            if success:
                frames.append(frame)

        capture.release()

        return frames

    def _process_frame(self, frame):
        """
        Detect, crop, align, and convert
        one video frame into a tensor.

        Pipeline:

            Frame
              ↓
            Face Detection
              ↓
            Largest Face
              ↓
            Face Crop
              ↓
            Face Alignment
              ↓
            RGB
              ↓
            Tensor
        """

        # ==================================================
        # 1. FACE DETECTION
        # ==================================================

        faces = self.face_detector.detect(
            frame
        )

        if not faces:
            return None

        # ==================================================
        # 2. SELECT LARGEST FACE
        # ==================================================
        #
        # If multiple faces are detected,
        # use the largest one.
        #
        # This is a simple subject-selection strategy
        # for the current version of the project.
        # ==================================================

        faces = sorted(
            faces,
            key=lambda box: box[2] * box[3],
            reverse=True,
        )

        largest_face = faces[0]

        # ==================================================
        # 3. FACE CROPPING
        # ==================================================

        cropped_faces = (
            self.face_processor.process_frame(
                frame,
                [largest_face],
            )
        )

        if not cropped_faces:
            return None

        cropped_face = cropped_faces[0]

        # ==================================================
        # 4. FACE ALIGNMENT
        # ==================================================

        aligned_face = self.face_aligner.align(
            cropped_face
        )

        # --------------------------------------------------
        # IMPORTANT FALLBACK
        # --------------------------------------------------
        #
        # MediaPipe may fail to find landmarks on some
        # frames even when the face detector successfully
        # detects a face.
        #
        # Instead of throwing away the frame, use the
        # cropped face.
        #
        # This keeps the dataset robust.
        # --------------------------------------------------

        if aligned_face is None:
            aligned_face = cropped_face

        # ==================================================
        # 5. BGR → RGB
        # ==================================================

        rgb_face = cv2.cvtColor(
            aligned_face,
            cv2.COLOR_BGR2RGB,
        )

        # ==================================================
        # 6. NUMPY → PYTORCH
        # ==================================================

        face_tensor = torch.from_numpy(
            rgb_face.copy()
        )

        # [H, W, C]
        #      ↓
        # [C, H, W]

        face_tensor = face_tensor.permute(
            2,
            0,
            1,
        )

        # ==================================================
        # 7. NORMALIZE PIXEL VALUES
        # ==================================================
        #
        # uint8 [0,255]
        #       ↓
        # float32 [0,1]
        # ==================================================

        face_tensor = (
            face_tensor.float() / 255.0
        )

        return face_tensor

    def __getitem__(self, index):
        """
        Return one processed video and its label.

        Returns:

            video_tensor:
                [num_frames, 3, 224, 224]

            label_tensor:
                0.0 = real
                1.0 = fake
        """

        video_path = self.video_paths[index]

        label = self.labels[index]

        # ==================================================
        # 1. SAMPLE VIDEO FRAMES
        # ==================================================

        frames = self._sample_frames(
            video_path
        )

        processed_frames = []

        # ==================================================
        # 2. PROCESS EACH FRAME
        # ==================================================

        for frame in frames:

            face_tensor = self._process_frame(
                frame
            )

            if face_tensor is not None:

                processed_frames.append(
                    face_tensor
                )

        # ==================================================
        # 3. MAKE SURE WE HAVE AT LEAST ONE FRAME
        # ==================================================

        if not processed_frames:

            raise RuntimeError(
                f"No usable face frames found in: "
                f"{video_path}"
            )

        # ==================================================
        # 4. HANDLE MISSING FRAMES
        # ==================================================
        #
        # Example:
        #
        # Requested:
        #     16 frames
        #
        # Successfully processed:
        #     13 frames
        #
        # Repeat the final valid frame:
        #
        #     13 → 14 → 15 → 16
        #
        # This guarantees a fixed temporal input size.
        # ==================================================

        while len(processed_frames) < self.num_frames:

            processed_frames.append(
                processed_frames[-1].clone()
            )

        # ==================================================
        # 5. KEEP EXACTLY num_frames
        # ==================================================

        processed_frames = (
            processed_frames[
                :self.num_frames
            ]
        )

        # ==================================================
        # 6. STACK FRAMES
        # ==================================================
        #
        # Individual:
        #     [3, 224, 224]
        #
        # Final:
        #     [16, 3, 224, 224]
        # ==================================================

        video_tensor = torch.stack(
            processed_frames
        )

        # ==================================================
        # 7. CREATE LABEL
        # ==================================================
        #
        # real = 0.0
        # fake = 1.0
        # ==================================================

        label_tensor = torch.tensor(
            label,
            dtype=torch.float32,
        )

        return video_tensor, label_tensor

    def close(self):
        """
        Release MediaPipe resources.
        """

        if self.face_aligner is not None:

            self.face_aligner.close()

            self.face_aligner = None

    def __del__(self):
        """
        Safely release resources when the dataset
        object is destroyed.
        """

        try:
            self.close()

        except Exception:
            pass