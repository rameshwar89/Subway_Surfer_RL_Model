from .transport import get_transport


class ScreenCapture:

    def __init__(self):
        self.transport = get_transport()

    def grab(self):
        return self.transport.get_frame()

    def close(self):
        pass