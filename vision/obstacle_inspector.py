import cv2


class ObstacleInspector:

    def __init__(self):
        pass

    def inspect(self, frame):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        # Smooth image
        blur = cv2.GaussianBlur(
            gray,
            (5, 5),
            0,
        )

        # Detect edges
        edges = cv2.Canny(
            blur,
            80,
            180,
        )

        # Find contours
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        display = frame.copy()

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < 400:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(
                display,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

        cv2.imshow(
            "Obstacle Inspector",
            display,
        )

        cv2.waitKey(1)