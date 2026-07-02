import cv2
import numpy as np


class ObservationProcessor:

    def __init__(self):

        # Output resolution
        self.width = 128
        self.height = 128

    def process(self, frame):

        # ---------------------------------------------------
        # 1. Convert RGB -> Grayscale
        # ---------------------------------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---------------------------------------------------
        # 2. Resize
        # ---------------------------------------------------
        obs = cv2.resize(
            gray,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA,
        )

        # ---------------------------------------------------
        # 3. Add channel dimension
        # Gym CNN expects H x W x C
        # ---------------------------------------------------
        obs = np.expand_dims(obs, axis=-1)

        return obs