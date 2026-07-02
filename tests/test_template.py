import cv2

from android.capture import ScreenCapture
from vision.template_matcher import TemplateMatcher

capture = ScreenCapture()

matcher = TemplateMatcher(
    "vision/templates/play.png",
    threshold=0.80,
)

while True:

    frame = capture.grab()

    found, score, location, size = matcher.detect(frame)

    display = cv2.resize(frame, (360, 800))

    sx = display.shape[1] / frame.shape[1]
    sy = display.shape[0] / frame.shape[0]

    if found:

        x, y = location
        w, h = size

        x = int(x * sx)
        y = int(y * sy)
        w = int(w * sx)
        h = int(h * sy)

        cv2.rectangle(
            display,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2,
        )

    text = f"{found} | {score:.3f}"

    cv2.putText(
        display,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Template Test", display)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()