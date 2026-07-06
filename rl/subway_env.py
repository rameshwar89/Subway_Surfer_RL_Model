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
from vision.live_observation import LiveObservationDebugger

PROFILE_ENV = False
DEBUG_OBSERVATION = True

STEP_DELAY = 0.10
WARMUP_FRAMES = 4
STARTUP_IGNORE_ACTIONS = 5
PROFILE_INTERVAL = 100
RESET_DELAY = 0.05
RESET_TAP_DELAY = 0.10
MACRO_POST_TAP_DELAY = 0.25
RESET_STATE_TIMEOUT = 3.0



class SubwayEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.capture = ScreenCapture()
        self.actions = SubwayActions()
        self.controller = GameController()
        self.detector = StateDetector()

        self.processor = ObservationProcessor()
        if DEBUG_OBSERVATION:
            self.live_debugger = LiveObservationDebugger()
        self.reward_system = RewardSystem()
        self.last_action_times = {}
        self.episode_steps = 0
        self.action_history = deque(maxlen=12)

        # ---------------------------------
        # Action Cooldowns (seconds).
        # Define physical durations of animations for cooldown checks.
        # ---------------------------------
        self.action_hold = {
            SubwayActions.LEFT: 0.40,
            SubwayActions.RIGHT: 0.40,
            SubwayActions.JUMP: 1.09,
            SubwayActions.ROLL: 0.50,
        }

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
            low=0,
            high=255,
            shape=(
                self.processor.height,
                self.processor.width,
                4,
            ),
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

        crop = frame[
            self.processor.crop_y:self.processor.crop_y + self.processor.crop_h,
            self.processor.crop_x:self.processor.crop_x + self.processor.crop_w,
        ]

        gray = self.processor.grayscale(crop)

        obs = self.processor.process(frame)

        self.frame_stack.append(obs)

        stacked = self._get_stacked_observation()

        return stacked, crop, gray

    # --------------------------------------------------
    # Profiling Helpers
    # --------------------------------------------------

    def _reset_profile(self):

        self.profile = {
            "action": 0.0,
            "sleep": 0.0,
            "capture": 0.0,
            "detect": 0.0,
            "process": 0.0,
            "reward": 0.0,
            "total": 0.0,
        }


    def _print_profile(self):

        if not PROFILE_ENV:
            return

        if self.profile_steps != PROFILE_INTERVAL:
            return

        print("\n========== ENV PROFILE ==========")
        print(f"Action Execute : {(self.profile['action']/PROFILE_INTERVAL)*1000:.2f} ms")
        print(f"Sleep          : {(self.profile['sleep']/PROFILE_INTERVAL)*1000:.2f} ms")
        print(f"Screen Capture : {(self.profile['capture']/PROFILE_INTERVAL)*1000:.2f} ms")
        print(f"State Detect   : {(self.profile['detect']/PROFILE_INTERVAL)*1000:.2f} ms")
        print(f"Preprocess     : {(self.profile['process']/PROFILE_INTERVAL)*1000:.2f} ms")
        print(f"Reward         : {(self.profile['reward']/PROFILE_INTERVAL)*1000:.2f} ms")
        print("--------------------------------")
        print(f"Total Step     : {(self.profile['total']/PROFILE_INTERVAL)*1000:.2f} ms")
        print(f"Steps / Second : {PROFILE_INTERVAL/self.profile['total']:.2f}")
        print("================================\n")

        self._reset_profile()
        self.profile_steps = 0

    def _build_transition(
        self,
        frame,
        action,
        reward,
        state,
        done,
        step_start,
        reward_breakdown=None,
    ):

        t = time.perf_counter()

        # Frame stack was updated during the blocking loop in step().
        # Compute crop/gray here solely for the debugger display.
        crop = frame[
            self.processor.crop_y:self.processor.crop_y + self.processor.crop_h,
            self.processor.crop_x:self.processor.crop_x + self.processor.crop_w,
        ]
        gray = self.processor.grayscale(crop)
        stacked = self._get_stacked_observation()

        self.profile["process"] += time.perf_counter() - t
        self.profile["total"] += time.perf_counter() - step_start

        if DEBUG_OBSERVATION:

            self.live_debugger.show(
                frame=frame,
                processor=self.processor,
                crop=crop,
                gray=gray,
                stacked=stacked,
                action=action,
                reward=reward,
                state=state,
                episode_steps=self.episode_steps,
                env_sps=(
                    PROFILE_INTERVAL / self.profile["total"]
                    if self.profile["total"] > 0
                    else 0
                ),
                reward_breakdown=reward_breakdown,
                detector_debug=self.detector.last_debug,
            )

        info = {
            "state": state,
            "action": action,
            "episode_steps": self.episode_steps,
            "step_time": time.perf_counter() - step_start,
            "reward_breakdown": reward_breakdown,
            "detector_debug": self.detector.last_debug,
        }

        self.profile_steps += 1

        self._print_profile()

        return (
            stacked,
            reward,
            done,
            False,
            info,
        )

    def _running_reward_breakdown(self, action):

        breakdown = {
            "survival": self.reward_system.SURVIVAL_REWARD,
            "time_bonus": min(
                self.episode_steps * 0.0003,
                0.03,
            ),
            "action_penalty": 0.0,
            "death": 0.0,
            "terminal_action_penalty": 0.0,
        }

        if action == SubwayActions.JUMP:
            breakdown["action_penalty"] = self.reward_system.JUMP_PENALTY

        elif action == SubwayActions.ROLL:
            breakdown["action_penalty"] = self.reward_system.ROLL_PENALTY

        return breakdown

    def _terminal_action_penalty(self):

        penalty = 0.0
        
        # Weights for the last 12 actions (oldest to newest)
        # Extending history window to 12 steps to capture earlier stumbles.
        weights = [0.02, 0.03, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.2, 0.15, 0.1, 0.1]
        
        # Align weights with the actual number of actions in history (in case episode was short)
        active_weights = weights[-len(self.action_history):] if self.action_history else []

        breakdown_strs = []
        
        if self.action_history:
            print("\n========== DEATH PENALTY DISTRIBUTION ==========")
            
        for i, action in enumerate(self.action_history):
            action_names_map = {0: "LEFT", 1: "RIGHT", 2: "JUMP", 3: "ROLL", 4: "IDLE"}
            action_name = action_names_map.get(int(action), str(action))
            step_pen = 0.0
            
            # Manually penalize all actions leading to death based on time weights
            if action == SubwayActions.JUMP:
                step_pen = -(1.0 * active_weights[i])
            elif action == SubwayActions.ROLL:
                step_pen = -(0.8 * active_weights[i])
            elif action in (SubwayActions.LEFT, SubwayActions.RIGHT):
                step_pen = -(0.9 * active_weights[i])
            elif action == SubwayActions.IDLE:
                step_pen = -(0.2 * active_weights[i])
                
            penalty += step_pen
            break_str = f"t-{len(self.action_history) - i - 1}: {action_name}({step_pen:+.2f})"
            breakdown_strs.append(break_str)
            print(f"{break_str} -> (Weight: {active_weights[i]:.1f})")

        if self.action_history:
            print(f"------------------------------------------------")
            print(f"TOTAL TERMINAL PENALTY: {penalty:+.2f}")
            print("================================================\n")

        return penalty, breakdown_strs

    def _game_over_reward(self, action, include_action_history_penalty):

        reward = self.reward_system.compute(
            state="GAME_OVER",
            action=action,
            episode_steps=self.episode_steps,
        )

        breakdown = {
            "survival": 0.0,
            "time_bonus": 0.0,
            "action_penalty": 0.0,
            "death": self.reward_system.GAME_OVER_PENALTY,
            "terminal_action_penalty": 0.0,
        }

        if include_action_history_penalty:
            pen, b_strs = self._terminal_action_penalty()
            breakdown["terminal_action_penalty"] = pen
            breakdown["terminal_strings"] = b_strs
            reward += pen

        return reward, breakdown

    def _detect_state(self, frame=None):

        if frame is None:
            frame = self.capture.grab()

        state, votes, scores = self.detector.detect(frame)

        return state, votes, scores, frame

    def _wait_for_state(self, target_state, timeout=RESET_STATE_TIMEOUT):

        deadline = time.perf_counter() + timeout

        while time.perf_counter() < deadline:

            state, _, _, _ = self._detect_state()

            if state == target_state:
                return True

            time.sleep(RESET_TAP_DELAY)

        return False

    # --------------------------------------------------
    # Gym API
    # --------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.frame_stack.clear()
        self.action_history.clear()
        self.reward_system.reset()
        self.episode_steps = 0
        self.detector.reset_episode()
        self.last_action_times.clear()

        # Simple architecture: spam tap_play until the pause button appears
        while True:
            if DEBUG_OBSERVATION:
                import cv2
                cv2.waitKey(1)

            frame = self.capture.grab()
            
            # Using context="gameplay" means we ONLY check the pause button
            state, _, _ = self.detector.detect(frame, context="gameplay")
            
            if state == "RUNNING" and self.detector.pause_missing_frames == 0:
                # Pause button is visible, game has started
                break
                
            # Pause button is missing, spam play
            self.controller.tap_play()
            time.sleep(0.01)
            
        # Give the game 2 seconds to stabilize (warmup) before passing control to PPO
        time.sleep(2.2)

        # No need for time-locked warmup because we specifically waited for the pause button to appear!
        self.frame_stack.clear()
        # Grab initial frames to fill the stack
        for _ in range(4):
            time.sleep(STEP_DELAY)
            frame = self.capture.grab()
            obs = self.processor.process(frame)
            self.frame_stack.append(obs)

        return self._get_stacked_observation(), {}

    def step(self, action):
        step_start = time.perf_counter()

        # Check for redundant action spamming
        redundant_penalty = 0.0
        if action != SubwayActions.IDLE:
            now = time.perf_counter()
            time_since_last = now - self.last_action_times.get(action, 0.0)
            cooldown = self.action_hold.get(action, 0.0)
            if time_since_last < cooldown:
                redundant_penalty = -0.80
            self.last_action_times[action] = now

        # Execute Action
        t = time.perf_counter()
        self.actions.execute(action)
        self.action_history.append(action)
        self.profile["action"] += time.perf_counter() - t

        # Grab a single frame for the transition
        t = time.perf_counter()
        frame = self.capture.grab()
        self.profile["capture"] += time.perf_counter() - t

        # Detect game state
        t = time.perf_counter()
        state, _, _ = self.detector.detect(frame, context="gameplay")
        self.profile["detect"] += time.perf_counter() - t

        # Process frame and append to observation history
        t = time.perf_counter()
        obs = self.processor.process(frame)
        self.frame_stack.append(obs)
        self.profile["process"] += time.perf_counter() - t

        # Determine terminal state and rewards
        if state == "USE_KEYS":
            t = time.perf_counter()
            reward, breakdown = self._game_over_reward(
                action=action,
                include_action_history_penalty=True,
            )
            # Apply redundant penalty if game ended on a spam step
            reward += redundant_penalty
            breakdown["action_penalty"] += redundant_penalty
            self.profile["reward"] += time.perf_counter() - t

            return self._build_transition(
                frame=frame,
                action=action,
                reward=reward,
                state="GAME_OVER",
                done=True,
                step_start=step_start,
                reward_breakdown=breakdown,
            )

        elif state in ("GAME_OVER_UI", "MAIN_MENU"):
            t = time.perf_counter()
            reward, breakdown = self._game_over_reward(
                action=action,
                include_action_history_penalty=True,
            )
            reward += redundant_penalty
            breakdown["action_penalty"] += redundant_penalty
            self.profile["reward"] += time.perf_counter() - t

            return self._build_transition(
                frame=frame,
                action=action,
                reward=reward,
                state="GAME_OVER",
                done=True,
                step_start=step_start,
                reward_breakdown=breakdown,
            )

        else:
            self.episode_steps += 1
            t = time.perf_counter()
            reward = self.reward_system.compute(
                state=state,
                action=action,
                episode_steps=self.episode_steps,
            )
            breakdown = self._running_reward_breakdown(action)
            
            # Apply redundant penalty
            reward += redundant_penalty
            breakdown["action_penalty"] += redundant_penalty
            self.profile["reward"] += time.perf_counter() - t

            # Enforce STEP_DELAY to prevent executing actions faster than the game runs
            elapsed = time.perf_counter() - step_start
            sleep_time = max(0, STEP_DELAY - elapsed)
            time.sleep(sleep_time)
            self.profile["sleep"] += sleep_time

            return self._build_transition(
                frame=frame,
                action=action,
                reward=reward,
                state="RUNNING",
                done=False,
                step_start=step_start,
                reward_breakdown=breakdown,
            )


    def close(self):

        if DEBUG_OBSERVATION:
            self.live_debugger.close()
