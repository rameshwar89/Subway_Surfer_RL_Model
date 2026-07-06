import json
import time

from android.transport import get_transport


class GameController:

    def __init__(self):

        self.transport = get_transport()

        with open("configs/calibration.json") as f:
            self.cfg = json.load(f)

    def tap(self, x, y):

        self.transport.tap(x, y)

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

    def tap_pause(self):

        self.tap(
            self.cfg["pause_x"],
            self.cfg["pause_y"],
        )

    def tap_quit(self):

        self.tap(
            self.cfg["quit_x"],
            self.cfg["quit_y"],
        )

    def tap_leave(self):

        self.tap(
            self.cfg["leave_x"],
            self.cfg["leave_y"],
        )
