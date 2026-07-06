import json

import cv2
import numpy as np


class LaneDetector:

    LEFT = 0
    CENTER = 1
    RIGHT = 2

    def __init__(self):

        with open("configs/lanes.json", "r") as f:
            cfg = json.load(f)

        self.left_top = np.array(cfg["left_top"], dtype=np.float32)
        self.left_bottom = np.array(cfg["left_bottom"], dtype=np.float32)

        self.right_top = np.array(cfg["right_top"], dtype=np.float32)
        self.right_bottom = np.array(cfg["right_bottom"], dtype=np.float32)

    def _interpolate(self, p1, p2, y):

        """
        Returns x-coordinate of the line at a given y.
        """

        x1, y1 = p1
        x2, y2 = p2

        if abs(y2 - y1) < 1e-6:
            return x1

        t = (y - y1) / (y2 - y1)

        return x1 + t * (x2 - x1)

    def draw(self, frame):

        h, w = frame.shape[:2]

        # Draw outer lane boundaries
        cv2.line(
            frame,
            tuple(self.left_top.astype(int)),
            tuple(self.left_bottom.astype(int)),
            (0, 255, 0),
            2,
        )

        cv2.line(
            frame,
            tuple(self.right_top.astype(int)),
            tuple(self.right_bottom.astype(int)),
            (0, 255, 0),
            2,
        )

        # Draw estimated center lane
        for y in range(0, h, 8):

            left_x = self._interpolate(
                self.left_top,
                self.left_bottom,
                y,
            )

            right_x = self._interpolate(
                self.right_top,
                self.right_bottom,
                y,
            )

            lane_width = (right_x - left_x) / 3.0

            x1 = int(left_x + lane_width)
            x2 = int(left_x + 2 * lane_width)

            cv2.circle(
                frame,
                (x1, y),
                1,
                (255, 255, 0),
                -1,
            )

            cv2.circle(
                frame,
                (x2, y),
                1,
                (255, 255, 0),
                -1,
            )

        return frame