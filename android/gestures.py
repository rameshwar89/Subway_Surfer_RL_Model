from .adb_controller import ADBController


class Gestures:

    def __init__(self):
        self.adb = ADBController()

    def swipe(self, x1, y1, x2, y2, duration=60):

        self.adb.shell(
            "input",
            "swipe",
            str(x1),
            str(y1),
            str(x2),
            str(y2),
            str(duration),
        )