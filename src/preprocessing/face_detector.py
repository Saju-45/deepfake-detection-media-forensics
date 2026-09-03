import cv2
import numpy as np


class FaceDetector:
    """Detect faces in video frames using OpenCV Haar Cascade."""

    def __init__(self):
        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self.detector = cv2.CascadeClassifier(cascade_path)

        if self.detector.empty():
            raise RuntimeError(
                "Failed to load Haar Cascade face detector."
            )

    def detect(self, frame):
        """Detect faces in a single frame."""

        if frame is None:
            raise ValueError("Input frame cannot be None.")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        return list(faces)

    def detect_and_crop(self, frame):
        """Detect faces and return cropped face images."""

        faces = self.detect(frame)

        cropped_faces = []

        for x, y, w, h in faces:
            face = frame[y:y + h, x:x + w]

            if face.size > 0:
                cropped_faces.append(face)

        return cropped_faces