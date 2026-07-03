import json

import cv2
import numpy as np


class ObservationProcessor:

    def __init__(self):

        self.width = 128
        self.height = 128

        with open("configs/observation.json", "r") as f:
            cfg = json.load(f)

        self.crop_x = cfg["crop_x"]
        self.crop_y = cfg["crop_y"]
        self.crop_w = cfg["crop_width"]
        self.crop_h = cfg["crop_height"]

    def process(self, frame):

        # -----------------------------
        # Crop gameplay
        # -----------------------------
        frame = frame[
            self.crop_y:self.crop_y + self.crop_h,
            self.crop_x:self.crop_x + self.crop_w,
        ]

        # -----------------------------
        # Convert to grayscale
        # -----------------------------
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        # -----------------------------
        # Resize
        # -----------------------------
        obs = cv2.resize(
            gray,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA,
        )

        # -----------------------------
        # Add channel dimension
        # -----------------------------
        return np.expand_dims(obs, axis=-1)