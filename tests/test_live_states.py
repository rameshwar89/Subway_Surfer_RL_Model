import cv2

from android.capture import ScreenCapture
from vision.state_detector import StateDetector

capture = ScreenCapture()
detector = StateDetector()

while True:

    frame = capture.grab()

    state, votes, scores = detector.detect(frame)

    display = cv2.resize(frame, (360, 800))

    cv2.putText(
        display,
        f"STATE : {state}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2,
    )

    cv2.putText(
        display,
        f"Votes : {votes}",
        (10,70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,0),
        2,
    )

    y = 110

    for name, score in scores.items():

        cv2.putText(
            display,
            f"{name}: {score:.3f}",
            (10,y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,255,255),
            1,
        )

        y += 22

    print(
        f"\rState: {state:<10} Votes:{votes} Scores:{scores}",
        end=""
    )

    cv2.imshow("Live State Detector", display)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()