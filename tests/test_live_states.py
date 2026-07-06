import cv2

from android.capture import ScreenCapture
from vision.state_detector import StateDetector

capture = ScreenCapture()
detector = StateDetector()

FONT = cv2.FONT_HERSHEY_SIMPLEX


def best_numeric_score(score_dict):

    numeric_scores = {
        name: score
        for name, score in score_dict.items()
        if isinstance(score, (int, float))
    }

    if not numeric_scores:
        return None, None

    best = max(
        numeric_scores,
        key=numeric_scores.get,
    )

    return best, numeric_scores[best]


def draw_panel(
    img,
    x,
    y,
    w,
    h,
    alpha=0.60,
):
    overlay = img.copy()

    cv2.rectangle(
        overlay,
        (x, y),
        (x + w, y + h),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        alpha,
        img,
        1.0 - alpha,
        0,
        img,
    )


while True:

    frame = capture.grab()

    state, votes, scores = detector.detect(frame)

    display = cv2.resize(frame, (360, 800))
    draw_panel(
        display,
        x=5,
        y=5,
        w=360,
        h=800,
        alpha=0.55,
    )

    # -----------------------------
    # Header
    # -----------------------------

    cv2.putText(
        display,
        f"STATE : {state}",
        (10, 30),
        FONT,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        display,
        f"Votes : {votes}",
        (10, 60),
        FONT,
        0.6,
        (255, 255, 255),
        2,
    )

    y = 95

    # -----------------------------
    # Detector Scores
    # -----------------------------

    cv2.putText(
        display,
        "Detector Scores",
        (10, y),
        FONT,
        0.55,
        (255, 255, 0),
        2,
    )

    y += 25

    for detector_name, detector_score in scores.items():

        if isinstance(detector_score, dict):

            best, score = best_numeric_score(detector_score)

            if best is not None:

                text = (
                    f"{detector_name:15}"
                    f"{score:.3f}"
                    f"   [{best}]"
                )

            else:

                text = f"{detector_name:15} none"

        else:

            if isinstance(detector_score, (int, float)):
                text = (
                    f"{detector_name:15}"
                    f"{detector_score:.3f}"
                )
            else:
                text = (
                    f"{detector_name:15}"
                    f"{detector_score}"
                )

        cv2.putText(
            display,
            text,
            (10, y),
            FONT,
            0.45,
            (255, 255, 255),
            1,
        )

        y += 20

    # -----------------------------
    # Loaded Templates
    # -----------------------------

    y += 10

    cv2.putText(
        display,
        "Loaded Templates",
        (10, y),
        FONT,
        0.55,
        (255, 255, 0),
        2,
    )

    y += 25

    template_info = [
        (
            "UseKeys",
            len(getattr(detector.use_keys_detector, "templates", []))
            if hasattr(detector, "use_keys_detector") else "-"
        ),
        (
            "GameOver",
            len(getattr(detector.game_over_detector, "templates", []))
            if hasattr(detector, "game_over_detector") else "-"
        ),
        (
            "MainMenu",
            len(getattr(detector.main_menu_detector, "templates", []))
            if hasattr(detector, "main_menu_detector") else "-"
        ),
        (
            "Pause",
            len(getattr(detector.pause_menu_detector, "templates", []))
            if hasattr(detector, "pause_menu_detector") else "-"
        ),
        (
            "Leave",
            len(getattr(detector.leave_confirm_detector, "templates", []))
            if hasattr(detector, "leave_confirm_detector") else "-"
        ),
    ]

    for name, count in template_info:

        cv2.putText(
            display,
            f"{name:10}: {count}",
            (10, y),
            FONT,
            0.45,
            (180, 255, 180),
            1,
        )

        y += 18

    # -----------------------------
    # Console Debug
    # -----------------------------

    print(
        f"\r"
        f"STATE={state:<15}"
        f" Votes={votes}",
        end="",
        flush=True,
    )

    cv2.imshow("Live State Detector", display)

    key = cv2.waitKey(1)

    if key == 27:
        break

cv2.destroyAllWindows()
