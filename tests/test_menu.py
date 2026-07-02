import cv2

from android.capture import ScreenCapture
from vision.menu_detector import MenuDetector

capture = ScreenCapture()
detector = MenuDetector()

while True:

    frame = capture.grab()

    found, score, _ = detector.detect(frame)

    display = cv2.resize(frame, (360, 800))

    cv2.putText(
        display,
        f"{found}   {score:.3f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Menu Detector", display)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()