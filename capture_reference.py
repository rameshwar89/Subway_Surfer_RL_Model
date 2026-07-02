import os
import cv2

from android.capture import ScreenCapture

SAVE_DIR = "vision/references"

os.makedirs(SAVE_DIR, exist_ok=True)

capture = ScreenCapture()

print("=" * 50)
print("Reference Screenshot Tool")
print("=" * 50)
print()
print("Controls:")
print("  r -> Save RUNNING screen")
print("  g -> Save GAME OVER screen")
print("  m -> Save MAIN MENU screen")
print("  q -> Quit")
print()

while True:

    frame = capture.grab()

    display = cv2.resize(frame, (360, 800))

    cv2.putText(
        display,
        "r:RUN  g:GAMEOVER  m:MENU  q:QUIT",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Capture References", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        cv2.imwrite(
            os.path.join(SAVE_DIR, "running.png"),
            frame,
        )
        print("Saved: running.png")

    elif key == ord("g"):
        cv2.imwrite(
            os.path.join(SAVE_DIR, "game_over.png"),
            frame,
        )
        print("Saved: game_over.png")

    elif key == ord("m"):
        cv2.imwrite(
            os.path.join(SAVE_DIR, "main_menu.png"),
            frame,
        )
        print("Saved: main_menu.png")

    elif key == ord("q"):
        break

cv2.destroyAllWindows()