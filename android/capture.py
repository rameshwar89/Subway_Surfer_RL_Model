import threading
import time

import scrcpy


class ScreenCapture:

    def __init__(self):

        self.latest_frame = None
        self.lock = threading.Lock()

        self.client = scrcpy.Client(
            max_width=720,
            bitrate=8000000,
            max_fps=30,
        )

        self.client.add_listener(
            scrcpy.EVENT_FRAME,
            self._on_frame,
        )

        self.client.start(
            threaded=True,
            daemon_threaded=True,
        )

        print("Starting Scrcpy Stream...")

        while True:

            with self.lock:
                if self.latest_frame is not None:
                    break

            time.sleep(0.02)

        print("Scrcpy Stream Ready!")

    def _on_frame(self, frame):

        if frame is None:
            return

        with self.lock:
            self.latest_frame = frame

    def grab(self):

        while True:

            with self.lock:

                if self.latest_frame is not None:
                    return self.latest_frame.copy()

            time.sleep(0.001)

    def close(self):

        self.capture.close()