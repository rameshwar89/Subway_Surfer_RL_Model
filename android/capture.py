import subprocess

import cv2
import numpy as np


class ScreenCapture:

    def grab(self):

        process = subprocess.Popen(
            ["adb", "exec-out", "screencap", "-p"],
            stdout=subprocess.PIPE,
        )

        image_bytes = process.stdout.read()

        image = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR,
        )

        return image