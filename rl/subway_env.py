import time

from android.capture import ScreenCapture
from controller.actions import SubwayActions
from controller.game_controller import GameController
from vision.state_detector import StateDetector


import gymnasium as gym
from gymnasium import spaces
import numpy as np

from rl.observation import ObservationProcessor
class SubwayEnv(gym.Env):

    def __init__(self):

        self.capture = ScreenCapture()
        self.actions = SubwayActions()
        self.controller = GameController()
        self.detector = StateDetector()

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(128, 128, 1),
            dtype=np.uint8,
        )

        self.processor = ObservationProcessor()

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

                obs = self.processor.process(frame)

                return obs, {}

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

                terminated = done
                truncated = False

                obs = self.processor.process(frame)

                return (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                )


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

                terminated = done
                truncated = False

                obs = self.processor.process(frame)

                return (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                )

            # -----------------------------
            # Unexpected state
            # -----------------------------
            else:

                time.sleep(0.05)


    def close(self):
        pass