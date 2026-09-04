import os

import cv2
import mediapipe as mp
import numpy as np


class FaceAligner:
    """Detect facial landmarks and align faces using eye positions."""

    def __init__(self, model_path="models/face_landmarker.task"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Face Landmarker model not found: {model_path}"
            )

        base_options = mp.tasks.BaseOptions(
            model_asset_path=model_path
        )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = (
            mp.tasks.vision.FaceLandmarker.create_from_options(
                options
            )
        )

    def get_landmarks(self, face):
        """Detect facial landmarks on a cropped face."""

        if face is None:
            raise ValueError("Face cannot be None.")

        if face.size == 0:
            raise ValueError("Face image cannot be empty.")

        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_face,
        )

        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        return result.face_landmarks[0]

    def align(self, face):
        """Align a face using the left and right eye landmarks."""

        landmarks = self.get_landmarks(face)

        if landmarks is None:
            return None

        height, width = face.shape[:2]

        # MediaPipe face landmark indices.
        left_eye = np.array([
            landmarks[33].x * width,
            landmarks[33].y * height,
        ])

        right_eye = np.array([
            landmarks[263].x * width,
            landmarks[263].y * height,
        ])

        delta_y = right_eye[1] - left_eye[1]
        delta_x = right_eye[0] - left_eye[0]

        angle = np.degrees(
            np.arctan2(delta_y, delta_x)
        )

        center = (
            (left_eye[0] + right_eye[0]) / 2,
            (left_eye[1] + right_eye[1]) / 2,
        )

        rotation_matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0,
        )

        aligned_face = cv2.warpAffine(
            face,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )

        return aligned_face

    def close(self):
        """Release MediaPipe resources."""

        self.landmarker.close()