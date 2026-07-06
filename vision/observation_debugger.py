import json
from collections import deque

import cv2  # type: ignore[reportMissingImports]
import numpy as np

from android.capture import ScreenCapture

PRINT_CONSOLE_SHAPES = False
WINDOW_W = 1024
WINDOW_H = 760


class ObservationDebugger:

    def __init__(self):

        self.capture = ScreenCapture()

        with open("configs/observation.json", "r") as f:
            cfg = json.load(f)

        self.crop_x = cfg["crop_x"]
        self.crop_y = cfg["crop_y"]
        self.crop_w = cfg["crop_width"]
        self.crop_h = cfg["crop_height"]

        self.output_size = 128

        self.frame_stack = deque(maxlen=4)

    # --------------------------------------------------
    # Processing
    # --------------------------------------------------

    def crop(self, frame):

        return frame[
            self.crop_y:self.crop_y + self.crop_h,
            self.crop_x:self.crop_x + self.crop_w,
        ]

    def grayscale(self, frame):

        return cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

    def resize(self, frame):

        return cv2.resize(
            frame,
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_AREA,
        )

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    def stack_preview(self):

        if len(self.frame_stack) == 0:

            return np.zeros(
                (self.output_size, self.output_size),
                dtype=np.uint8,
            )

        imgs = list(self.frame_stack)

        while len(imgs) < 4:
            imgs.insert(0, imgs[0])

        top = np.hstack((imgs[0], imgs[1]))
        bottom = np.hstack((imgs[2], imgs[3]))

        return np.vstack((top, bottom))

    def annotate(self, image, text):

        cv2.putText(
            image,
            text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    # --------------------------------------------------
    # Main Loop
    # --------------------------------------------------

    def run(self):

        print("=" * 60)
        print("Observation Debugger")
        print("=" * 60)
        print("Press Q to quit.\n")

        cv2.namedWindow(
            "Observation Debugger",
            cv2.WINDOW_NORMAL,
        )
        cv2.setWindowProperty(
            "Observation Debugger",
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_NORMAL,
        )
        cv2.resizeWindow(
            "Observation Debugger",
            WINDOW_W,
            WINDOW_H,
        )
        cv2.moveWindow(
            "Observation Debugger",
            80,
            80,
        )

        while True:

            frame = self.capture.grab()

            original = frame.copy()

            # Draw crop rectangle
            cv2.rectangle(
                original,
                (self.crop_x, self.crop_y),
                (
                    self.crop_x + self.crop_w,
                    self.crop_y + self.crop_h,
                ),
                (0, 255, 0),
                3,
            )

            cropped = self.crop(frame)

            gray = self.grayscale(cropped)

            resized = self.resize(gray)

            self.frame_stack.append(resized)

            stack = self.stack_preview()

            # -----------------------------
            # Visualization
            # -----------------------------

            original_view = cv2.resize(
                original,
                (1024, 520),
            )

            crop_view = cv2.resize(
                cropped,
                (256, 256),
            )

            gray_view = cv2.resize(
                gray,
                (256, 256),
            )

            resized_view = cv2.resize(
                resized,
                (256, 256),
                interpolation=cv2.INTER_NEAREST,
            )

            stack_view = cv2.resize(
                stack,
                (256, 256),
                interpolation=cv2.INTER_NEAREST,
            )

            gray_view = cv2.cvtColor(
                gray_view,
                cv2.COLOR_GRAY2BGR,
            )

            resized_view = cv2.cvtColor(
                resized_view,
                cv2.COLOR_GRAY2BGR,
            )

            stack_view = cv2.cvtColor(
                stack_view,
                cv2.COLOR_GRAY2BGR,
            )

            self.annotate(crop_view, "Crop")
            self.annotate(gray_view, "Gray")
            self.annotate(resized_view, "128x128")
            self.annotate(stack_view, "Frame Stack")

            top = np.hstack((
                crop_view,
                gray_view,
                resized_view,
                stack_view,
            ))

            dashboard = np.vstack((
                original_view,
                top,
            ))

            cv2.imshow(
                "Observation Debugger",
                dashboard,
            )

            # -----------------------------
            # Console Info
            # -----------------------------

            if PRINT_CONSOLE_SHAPES:
                print(
                    f"\r"
                    f"Original: {frame.shape} | "
                    f"Crop: {cropped.shape} | "
                    f"Gray: {gray.shape} | "
                    f"Output: {resized.shape} | "
                    f"Stack: {len(self.frame_stack)}/4",
                    end="",
                )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":

    ObservationDebugger().run()
