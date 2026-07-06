import json

import cv2
import numpy as np


class ObservationProcessor:
    """
    Handles all observation preprocessing for PPO.

    Pipeline:
        Full Screenshot
            ↓
        Crop Gameplay
            ↓
        Grayscale
            ↓
        Resize
            ↓
        Normalize
            ↓
        (H, W, 1)
    """

    def __init__(self):

        with open("configs/observation.json", "r") as f:
            cfg = json.load(f)

        # Crop ROI
        self.crop_x = cfg["crop_x"]
        self.crop_y = cfg["crop_y"]
        self.crop_w = cfg["crop_width"]
        self.crop_h = cfg["crop_height"]

        # Output resolution
        self.width = cfg.get("resize_width", 128)
        self.height = cfg.get("resize_height", 128)

    def crop(self, frame):
        """Crop gameplay region."""

        h, w = frame.shape[:2]

        x1 = max(0, self.crop_x)
        y1 = max(0, self.crop_y)
        x2 = min(w, self.crop_x + self.crop_w)
        y2 = min(h, self.crop_y + self.crop_h)

        return frame[y1:y2, x1:x2]

    def grayscale(self, frame):

        return cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

    def resize(self, frame):
        """Resize observation."""

        return cv2.resize(
            frame,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA,
        )

    def normalize(self, frame):
        """
        Keep uint8 image.
        Stable-Baselines3 will normalize internally.
        """
        return frame

    def process(self, frame):

        frame = self.crop(frame)
        frame = self.grayscale(frame)
        frame = self.resize(frame)
        return np.expand_dims(frame, axis=-1)

    @property
    def observation_shape(self):
        return (self.height, self.width, 1)

    @property
    def observation_dtype(self):
        return np.uint8