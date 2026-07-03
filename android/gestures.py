from .transport import get_transport


class Gestures:

    def __init__(self):
        self.transport = get_transport()

    def swipe(self, x1, y1, x2, y2, duration=60):

        self.transport.swipe(
            x1,
            y1,
            x2,
            y2,
        )