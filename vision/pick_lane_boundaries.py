import json

import cv2

from android.capture import ScreenCapture


class LanePicker:

    def __init__(self):

        self.capture = ScreenCapture()

        with open("assets/configs/observation.json", "r") as f:
            cfg = json.load(f)

        self.crop_x = cfg["crop_x"]
        self.crop_y = cfg["crop_y"]
        self.crop_w = cfg["crop_width"]
        self.crop_h = cfg["crop_height"]

        self.points = []

    def mouse_callback(self, event, x, y, flags, param):

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        self.points.append((x, y))

        print(f"Point {len(self.points)} : ({x}, {y})")

    def run(self):

        frame = self.capture.grab()

        frame = frame[
            self.crop_y:self.crop_y + self.crop_h,
            self.crop_x:self.crop_x + self.crop_w,
        ]

        display = frame.copy()

        cv2.namedWindow("Lane Calibration")
        cv2.setMouseCallback(
            "Lane Calibration",
            self.mouse_callback,
        )

        print()
        print("====================================")
        print("Click in this order:")
        print()
        print("1. LEFT boundary TOP")
        print("2. LEFT boundary BOTTOM")
        print("3. RIGHT boundary TOP")
        print("4. RIGHT boundary BOTTOM")
        print()
        print("Press S to save.")
        print("Press Q to quit.")
        print("====================================")

        while True:

            img = display.copy()

            for p in self.points:

                cv2.circle(
                    img,
                    p,
                    6,
                    (0, 0, 255),
                    -1,
                )

            if len(self.points) >= 2:

                cv2.line(
                    img,
                    self.points[0],
                    self.points[1],
                    (0, 255, 0),
                    2,
                )

            if len(self.points) >= 4:

                cv2.line(
                    img,
                    self.points[2],
                    self.points[3],
                    (255, 0, 0),
                    2,
                )

            cv2.imshow(
                "Lane Calibration",
                img,
            )

            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                self.points.clear()

            if key == ord("s"):

                if len(self.points) != 4:

                    print("Need exactly 4 points.")
                    continue

                data = {
                    "left_top": list(self.points[0]),
                    "left_bottom": list(self.points[1]),
                    "right_top": list(self.points[2]),
                    "right_bottom": list(self.points[3]),
                }

                with open(
                    "assets/configs/lanes.json",
                    "w",
                ) as f:

                    json.dump(
                        data,
                        f,
                        indent=4,
                    )

                print("\nSaved lanes.json")
                print(json.dumps(data, indent=4))

                break

        cv2.destroyAllWindows()


if __name__ == "__main__":

    LanePicker().run()