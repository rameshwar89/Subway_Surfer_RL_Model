import cv2


class VisionLab:

    """
    Interactive vision playground.

    Press:

    1 -> Original
    2 -> Gray
    3 -> Blur
    4 -> Canny
    5 -> Difference (future)
    6 -> Contours (future)
    """

    ORIGINAL = 1
    GRAY = 2
    BLUR = 3
    CANNY = 4
    DIFFERENCE = 5
    CONTOURS = 6

    def __init__(self):

        self.mode = self.ORIGINAL

    def process(self, frame):

        key = cv2.waitKey(1) & 0xFF

        if key == ord("1"):
            self.mode = self.ORIGINAL

        elif key == ord("2"):
            self.mode = self.GRAY

        elif key == ord("3"):
            self.mode = self.BLUR

        elif key == ord("4"):
            self.mode = self.CANNY

        elif key == ord("5"):
            self.mode = self.DIFFERENCE

        elif key == ord("6"):
            self.mode = self.CONTOURS

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        blur = cv2.GaussianBlur(
            gray,
            (5, 5),
            0,
        )

        canny = cv2.Canny(
            blur,
            80,
            180,
        )

        if self.mode == self.ORIGINAL:

            display = frame

        elif self.mode == self.GRAY:

            display = gray

        elif self.mode == self.BLUR:

            display = blur

        elif self.mode == self.CANNY:

            display = canny

        else:

            display = frame

        cv2.imshow(
            "Vision Lab",
            display,
        )

        return display