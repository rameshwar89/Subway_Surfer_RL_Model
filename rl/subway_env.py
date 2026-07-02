import time
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from android.capture import ScreenCapture
from controller.actions import SubwayActions
from controller.game_controller import GameController
from rl.observation import ObservationProcessor
from rl.reward import RewardSystem
from vision.state_detector import StateDetector


class SubwayEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.capture = ScreenCapture()
        self.actions = SubwayActions()
        self.controller = GameController()
        self.detector = StateDetector()

        self.processor = ObservationProcessor()
        self.reward_system = RewardSystem()

        # Last 4 processed frames
        self.frame_stack = deque(maxlen=4)

        # RL Spaces
        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(128, 128, 4),
            dtype=np.uint8,
        )

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _get_stacked_observation(self):

        return np.concatenate(
            list(self.frame_stack),
            axis=-1,
        )

    def _update_observation(self, frame):

        obs = self.processor.process(frame)

        self.frame_stack.append(obs)

        return self._get_stacked_observation()

    # --------------------------------------------------
    # Gym API
    # --------------------------------------------------

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.frame_stack.clear()

        print("\n========== RESET ==========")

        self.controller.tap_play()

        while True:

            frame = self.capture.grab()

            state, _, _ = self.detector.detect(frame)

            print("RESET STATE:", state)

            if state == "GAME_OVER":

                self.controller.tap_play()

                time.sleep(0.5)

                continue

            elif state == "MAIN_MENU":

                self.controller.tap_start()

                time.sleep(0.5)

                continue

            elif state == "USE_KEYS":

                print("USE_KEYS detected -> Closing popup")

                self.controller.close_use_keys()

                time.sleep(0.5)

                continue

            elif state == "RUNNING":

                obs = self.processor.process(frame)

                # Fill stack with first observation
                for _ in range(4):
                    self.frame_stack.append(obs)

                return self._get_stacked_observation(), {}

            time.sleep(0.05)

    def step(self, action):

        # Execute action
        self.actions.execute(action)

        time.sleep(0.08)

        while True:

            frame = self.capture.grab()

            state, _, _ = self.detector.detect(frame)

            print(f"[STEP] State = {state}")

            # -----------------------------
            # Continue Episode
            # -----------------------------
            if state == "RUNNING":

                reward = self.reward_system.compute(state)

                info = {
                    "state": state,
                }

                stacked = self._update_observation(frame)

                return (
                    stacked,
                    reward,
                    False,      # terminated
                    False,      # truncated
                    info,
                )

            # -----------------------------
            # Revive Popup
            # -----------------------------
            elif state == "USE_KEYS":

                print("USE_KEYS detected -> Closing popup")

                self.controller.close_use_keys()

                time.sleep(0.5)

                continue

            # -----------------------------
            # Episode End
            # -----------------------------
            elif state == "GAME_OVER":

                reward = self.reward_system.compute(state)

                info = {
                    "state": state,
                }

                stacked = self._update_observation(frame)

                return (
                    stacked,
                    reward,
                    True,       # terminated
                    False,      # truncated
                    info,
                )

            # -----------------------------
            # Unknown State
            # -----------------------------
            else:

                time.sleep(0.05)

    def close(self):

        pass