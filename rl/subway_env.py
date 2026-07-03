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

PROFILE_ENV = False

STEP_DELAY = 0.08
WARMUP_FRAMES = 4
STARTUP_IGNORE_ACTIONS = 3
PROFILE_INTERVAL = 100
RESET_DELAY = 0.05


class SubwayEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.capture = ScreenCapture()
        self.actions = SubwayActions()
        self.controller = GameController()
        self.detector = StateDetector()

        self.processor = ObservationProcessor()
        self.reward_system = RewardSystem()
        self.startup_steps = 0
        self.episode_steps = 0

        # --------------------------
        # Performance Profiling
        # --------------------------

        self.profile = {
            "action": 0.0,
            "sleep": 0.0,
            "capture": 0.0,
            "detect": 0.0,
            "process": 0.0,
            "reward": 0.0,
            "total": 0.0,
        }

        self.profile_steps = 0

        # Last 4 processed frames
        self.frame_stack = deque(maxlen=4)

        # RL Spaces
        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(
                self.processor.height,
                self.processor.width,
                4,
            ),
            dtype=np.float32,
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

        self.reward_system.reset()

        self.episode_steps = 0
        self.startup_steps = 0

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

            elif state == "USE_KEYS":

                self.controller.close_use_keys()

                time.sleep(0.5)

                continue

            elif state == "RUNNING":

                # --------------------------------------------------
                # Warm-up after entering gameplay
                #
                # Instead of stacking the exact same frame 4 times,
                # wait for a few fresh gameplay frames so the agent
                # starts from a stable running state.
                # --------------------------------------------------

                self.frame_stack.clear()

                for _ in range(WARMUP_FRAMES):

                    time.sleep(RESET_DELAY)

                    frame = self.capture.grab()

                    obs = self.processor.process(frame)

                    self.frame_stack.append(obs)

                self.startup_steps = STARTUP_IGNORE_ACTIONS
                return self._get_stacked_observation(), {}
            time.sleep(RESET_DELAY)

    def step(self, action):
        # ---------------------------------
        # Startup warm-up
        # Ignore the first few PPO actions
        # after every reset.
        # ---------------------------------

        if self.startup_steps > 0:
            action = SubwayActions.IDLE
            self.startup_steps -= 1

        step_start = time.perf_counter()

        # ---------------------------------
        # Execute Action
        # ---------------------------------

        t = time.perf_counter()
        self.actions.execute(action)
        self.profile["action"] += time.perf_counter() - t

        # ---------------------------------
        # Wait for game to react
        # ---------------------------------

        t = time.perf_counter()
        time.sleep(STEP_DELAY)
        self.profile["sleep"] += time.perf_counter() - t

        while True:

            # ---------------------------------
            # Capture Screen
            # ---------------------------------

            t = time.perf_counter()
            frame = self.capture.grab()
            self.profile["capture"] += time.perf_counter() - t

            # ---------------------------------
            # Detect Game State
            # ---------------------------------

            t = time.perf_counter()
            state, _, _ = self.detector.detect(frame)
            self.profile["detect"] += time.perf_counter() - t

            # print(f"[STEP] State = {state}")

            # -----------------------------
            # Continue Episode
            # -----------------------------
            if state == "RUNNING":

                # Reward
                t = time.perf_counter()
                self.episode_steps += 1

                reward = self.reward_system.compute(
                    state=state,
                    action=action,
                    episode_steps=self.episode_steps,
                )
                self.profile["reward"] += time.perf_counter() - t

                # Observation Processing
                t = time.perf_counter()
                stacked = self._update_observation(frame)
                self.profile["process"] += time.perf_counter() - t

                info = {
                    "state": state,
                    "action": action,
                    "episode_steps": self.episode_steps,
                    "step_time": time.perf_counter() - step_start,
                    "reward_breakdown": None,  # placeholder for future versions
                }

                self.profile["total"] += time.perf_counter() - step_start
                self.profile_steps += 1

                if PROFILE_ENV and self.profile_steps == PROFILE_INTERVAL:

                    print("\n========== ENV PROFILE (100 Steps) ==========")
                    print(f"Action Execute : {(self.profile['action']/PROFILE_INTERVAL)*1000:.2f} ms")
                    print(f"Sleep          : {(self.profile['sleep']/PROFILE_INTERVAL)*1000:.2f} ms")
                    print(f"Screen Capture : {(self.profile['capture']/PROFILE_INTERVAL)*1000:.2f} ms")
                    print(f"State Detect   : {(self.profile['detect']/PROFILE_INTERVAL)*1000:.2f} ms")
                    print(f"Preprocess     : {(self.profile['process']/PROFILE_INTERVAL)*1000:.2f} ms")
                    print(f"Reward         : {(self.profile['reward']/PROFILE_INTERVAL)*1000:.2f} ms")
                    print("---------------------------------------------")
                    print(f"Total Step     : {(self.profile['total']/PROFILE_INTERVAL)*1000:.2f} ms")
                    print(f"Steps / Second : {PROFILE_INTERVAL/self.profile['total']:.2f}")
                    print("=============================================\n")

                    self.profile = {
                        "action": 0.0,
                        "sleep": 0.0,
                        "capture": 0.0,
                        "detect": 0.0,
                        "process": 0.0,
                        "reward": 0.0,
                        "total": 0.0,
                    }

                    self.profile_steps = 0

                return (
                    stacked,
                    reward,
                    False,
                    False,
                    info,
                )

            # -----------------------------
            # Revive Popup
            # -----------------------------
            elif state == "USE_KEYS":

                self.controller.close_use_keys()

                time.sleep(0.5)

                continue

            # -----------------------------
            # Episode End
            # -----------------------------
            elif state == "GAME_OVER":

                t = time.perf_counter()
                reward = self.reward_system.compute(
                    state=state,
                    action=action,
                    episode_steps=self.episode_steps,
                )
                self.profile["reward"] += time.perf_counter() - t

                t = time.perf_counter()
                stacked = self._update_observation(frame)
                self.profile["process"] += time.perf_counter() - t

                info = {
                    "state": state,
                    "action": action,
                    "episode_steps": self.episode_steps,
                    "step_time": time.perf_counter() - step_start,
                    "reward_breakdown": None,  # placeholder for future versions
                }

                self.profile["total"] += time.perf_counter() - step_start
                self.profile_steps += 1

                if PROFILE_ENV and self.profile_steps == PROFILE_INTERVAL:

                    print("\n========== ENV PROFILE (100 Steps) ==========")
                    print(f"Action Execute : {(self.profile['action']/PROFILE_INTERVAL)*1000:.2f} ms")
                    print(f"Sleep          : {(self.profile['sleep']/PROFILE_INTERVAL)*1000:.2f} ms")
                    print(f"Screen Capture : {(self.profile['capture']/PROFILE_INTERVAL)*1000:.2f} ms")
                    print(f"State Detect   : {(self.profile['detect']/PROFILE_INTERVAL)*1000:.2f} ms")
                    print(f"Preprocess     : {(self.profile['process']/PROFILE_INTERVAL)*1000:.2f} ms")
                    print(f"Reward         : {(self.profile['reward']/PROFILE_INTERVAL)*1000:.2f} ms")
                    print("---------------------------------------------")
                    print(f"Total Step     : {(self.profile['total']/PROFILE_INTERVAL)*1000:.2f} ms")
                    print(f"Steps / Second : {PROFILE_INTERVAL/self.profile['total']:.2f}")
                    print("=============================================\n")

                    self.profile = {
                        "action": 0.0,
                        "sleep": 0.0,
                        "capture": 0.0,
                        "detect": 0.0,
                        "process": 0.0,
                        "reward": 0.0,
                        "total": 0.0,
                    }

                    self.profile_steps = 0

                return (
                    stacked,
                    reward,
                    True,
                    False,
                    info,
                )

            # -----------------------------
            # Unknown State
            # -----------------------------
            else:
                t = time.perf_counter()
                time.sleep(RESET_DELAY)
                self.profile["sleep"] += time.perf_counter() - t
    
    def close(self):

        pass