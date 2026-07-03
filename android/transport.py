import threading
import time

import scrcpy


class AndroidTransport:
    """
    Shared persistent scrcpy connection.

    Owns:
        - Video stream
        - Touch controls
    """

    def __init__(self):

        self.lock = threading.Lock()
        self.latest_frame = None

        self.client = scrcpy.Client(
            bitrate=8_000_000,
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
        time.sleep(1)

        print("Starting Android Transport...")

        while self.latest_frame is None:
            time.sleep(0.02)

        print("Android Transport Ready!")

    def _on_frame(self, frame):

        if frame is None:
            return

        with self.lock:
            self.latest_frame = frame

    # -----------------------------
    # Video
    # -----------------------------

    def get_frame(self):

        while True:

            with self.lock:

                if self.latest_frame is not None:
                    return self.latest_frame.copy()

            time.sleep(0.001)
    # ---------------------------------
    # Touch / Swipe
    # ---------------------------------

    def swipe(
        self,
        x1,
        y1,
        x2,
        y2,
        move_step_length=50,
        move_steps_delay=0.0,
    ):
        self.client.control.swipe(
            x1,
            y1,
            x2,
            y2,
            move_step_length,
            move_steps_delay,
        )

    def stop(self):

        self.client.stop()

_transport = None


def get_transport():
    global _transport

    if _transport is None:
        _transport = AndroidTransport()

    return _transport