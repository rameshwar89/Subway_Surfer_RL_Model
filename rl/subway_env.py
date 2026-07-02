import time

from android.capture import ScreenCapture
from controller.actions import SubwayActions
from controller.game_controller import GameController
from vision.state_detector import StateDetector


class SubwayEnv:

    def __init__(self):

        self.capture = ScreenCapture()
        self.actions = SubwayActions()
        self.controller = GameController()
        self.detector = StateDetector()

    def reset(self):

        print("\n========== RESET ==========")

        self.controller.tap_play()

        while True:

            frame = self.capture.grab()

            state, _, _ = self.detector.detect(frame)

            if state == "GAME_OVER":

                self.controller.tap_play()

                time.sleep(0.5)

                continue

            elif state == "MAIN_MENU":

                self.controller.tap_start()

                time.sleep(0.5)

                continue

            elif state == "RUNNING":

                return frame

            time.sleep(0.05)

    def step(self, action):

        # Execute action
        self.actions.execute(action)

        time.sleep(0.08)

        while True:

            frame = self.capture.grab()

            state, _, _ = self.detector.detect(frame)

            # -----------------------------
            # Normal gameplay
            # -----------------------------
            if state == "RUNNING":

                reward = 1

                done = False

                info = {
                    "state": state,
                }

                return frame, reward, done, info

            # -----------------------------
            # Revive popup
            # -----------------------------
            elif state == "USE_KEYS":

                print("USE_KEYS detected -> Closing popup")

                self.controller.close_use_keys()

                time.sleep(0.5)

                continue

            # -----------------------------
            # Episode finished
            # -----------------------------
            elif state == "GAME_OVER":

                reward = -100

                done = True

                info = {
                    "state": state,
                }

                return frame, reward, done, info

            # -----------------------------
            # Unexpected state
            # -----------------------------
            else:

                time.sleep(0.05)