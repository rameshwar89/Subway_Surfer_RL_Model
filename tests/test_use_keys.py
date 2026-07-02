import time

from android.capture import ScreenCapture
from controller.game_controller import GameController
from vision.state_detector import StateDetector

capture = ScreenCapture()
controller = GameController()
detector = StateDetector()

print("=" * 60)
print("USE KEYS TEST")
print("=" * 60)
print("Instructions:")
print("1. Start Subway Surfers manually.")
print("2. Crash intentionally.")
print("3. DO NOT TOUCH ANYTHING after crashing.")
print("=" * 60)

last_state = None

while True:

    frame = capture.grab()

    state, votes, scores = detector.detect(frame)

    if state != last_state:

        print("\n--------------------------------")
        print(f"STATE : {state}")
        print(f"VOTES : {votes}")

        for name, score in scores.items():
            print(f"{name:20s}: {score:.3f}")

        last_state = state

    if state == "USE_KEYS":

        print("\n>>> USE KEYS DETECTED <<<")
        print("Closing popup...")

        controller.close_use_keys()

        time.sleep(1)

        continue

    if state == "GAME_OVER":

        print("\nWaiting at GAME OVER...")

    time.sleep(0.05)