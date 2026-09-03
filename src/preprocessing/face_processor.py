import cv2
import numpy as np


class FaceProcessor:
    """Crop and standardize detected face regions."""

    def __init__(self, output_size=(224, 224), margin=0.2):
        self.output_size = output_size
        self.margin = margin

    def crop_face(self, frame, bounding_box):
        """
        Crop a face from a frame using a bounding box.

        Parameters
        ----------
        frame : numpy.ndarray
            Original BGR video frame.

        bounding_box : tuple
            Face bounding box in (x, y, width, height) format.

        Returns
        -------
        numpy.ndarray
            Cropped and resized face.
        """

        if frame is None:
            raise ValueError("Input frame cannot be None.")

        if bounding_box is None:
            raise ValueError("Bounding box cannot be None.")

        x, y, w, h = bounding_box

        if w <= 0 or h <= 0:
            raise ValueError("Bounding box dimensions must be positive.")

        frame_height, frame_width = frame.shape[:2]

        # Add a small margin around the detected face.
        margin_x = int(w * self.margin)
        margin_y = int(h * self.margin)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)

        x2 = min(frame_width, x + w + margin_x)
        y2 = min(frame_height, y + h + margin_y)

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            raise ValueError("Cropped face is empty.")

        # Resize AFTER cropping to preserve the original face proportions.
        face = cv2.resize(face, self.output_size)

        return face

    def process_frame(self, frame, bounding_boxes):
        """
        Crop all detected faces from one frame.

        Parameters
        ----------
        frame : numpy.ndarray
            Original video frame.

        bounding_boxes : list
            List of face bounding boxes.

        Returns
        -------
        list
            List of processed face images.
        """

        processed_faces = []

        for bounding_box in bounding_boxes:
            face = self.crop_face(frame, bounding_box)
            processed_faces.append(face)

        return processed_faces