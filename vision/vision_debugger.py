import cv2

from vision.lane_detector import LaneDetector


class VisionDebugger:

    def __init__(self, enabled=True):

        self.enabled = enabled

        self.lane_detector = LaneDetector()

    def show(
        self,
        frame,
        lane="UNKNOWN",
    ):

        if not self.enabled:
            return

        display = frame.copy()

        # ---------------------------------------
        # Draw calibrated lane geometry
        # ---------------------------------------

        display = self.lane_detector.draw(display)

        # ---------------------------------------
        # Display current lane text
        # ---------------------------------------

        cv2.putText(
            display,
            f"Lane : {lane}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        cv2.imshow(
            "Vision Debugger",
            display,
        )

        cv2.waitKey(1)