import json

from android.gestures import Gestures


class SubwayActions:

    LEFT = 0
    RIGHT = 1
    JUMP = 2
    ROLL = 3
    IDLE = 4

    def __init__(self):

        self.g = Gestures()

        with open("configs/calibration.json") as f:
            self.cfg = json.load(f)

    def _swipe(self, x2, y2):

        self.g.swipe(
            self.cfg["center_x"],
            self.cfg["center_y"],
            x2,
            y2,
            self.cfg["swipe_duration"],
        )

    def execute(self, action):

        if action == self.LEFT:

            self._swipe(
                self.cfg["left_x"],
                self.cfg["center_y"],
            )

        elif action == self.RIGHT:

            self._swipe(
                self.cfg["right_x"],
                self.cfg["center_y"],
            )

        elif action == self.JUMP:

            self._swipe(
                self.cfg["center_x"],
                self.cfg["jump_y"],
            )

        elif action == self.ROLL:

            self._swipe(
                self.cfg["center_x"],
                self.cfg["roll_y"],
            )

        elif action == self.IDLE:

            pass