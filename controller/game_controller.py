import json
import time

from android.adb_controller import ADBController


class GameController:

    def __init__(self):

        self.adb = ADBController()

        with open("configs/calibration.json") as f:
            self.cfg = json.load(f)

    def tap(self, x, y):

        self.adb.shell(
            "input",
            "tap",
            str(x),
            str(y),
        )

    def tap_play(self):

        self.tap(
            self.cfg["play_x"],
            self.cfg["play_y"],
        )

    def close_use_keys(self):

        self.tap(
            self.cfg["close_use_keys_x"],
            self.cfg["close_use_keys_y"],
        )

    def tap_start(self):

        self.tap(
            self.cfg["start_x"],
            self.cfg["start_y"],
        )