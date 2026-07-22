import time
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from android.capture import ScreenCapture
from controller.actions import SubwayActions
from controller.game_controller import GameController
from rl.reward import RewardSystem
from vision.object_tracker import SubwayObjectTracker
from vision.state_vector_builder import StateVectorBuilder
from vision.state_detector import StateDetector
from vision.live_observation import LiveObservationDebugger
from vision.lane_detector import LaneDetector

PROFILE_ENV = False
DEBUG_OBSERVATION = True
STEP_DELAY = 0.20
WARMUP_FRAMES = 4
STARTUP_IGNORE_ACTIONS = 5
PROFILE_INTERVAL = 100
RESET_DELAY = 0.05
RESET_TAP_DELAY = 0.10
MACRO_POST_TAP_DELAY = 0.25
RESET_STATE_TIMEOUT = 3.0

# Stumble detection tuning (removed)



class SubwayEnv(gym.Env):

    def __init__(self, initial_episode=0):

        super().__init__()

        self.capture = ScreenCapture()
        self.actions = SubwayActions()
        self.controller = GameController()
        self.detector = StateDetector()

        self.tracker = SubwayObjectTracker()
        self.vector_builder = StateVectorBuilder()
        self.lane_detector = LaneDetector()
        self.current_agent_lane = 1 # Start in center lane

        if DEBUG_OBSERVATION:
            self.live_debugger = LiveObservationDebugger(initial_episode)
        self.reward_system = RewardSystem()
        self.last_action_times = {}
        self.episode_steps = 0
        self.action_history = deque(maxlen=5)
        self.on_train = False
        self.on_train_steps = 0
        self.total_env_steps = 0

        # Stumble detection state
        self.stumble_immunity   = 0   # Countdown: no stumble detection while > 0
        self.stumble_window     = 0   # Countdown: stumble valid while > 0 (after lane action)
        self.stumble_revert_to  = 1   # Lane to restore if stumble confirmed
        self.police_consecutive = 0   # Consecutive frames police has been present

        # ---------------------------------
        # Action Cooldowns (seconds).
        # Define physical durations of animations for cooldown checks.
        # ---------------------------------
        self.action_hold = {
            SubwayActions.LEFT: 0.35,
            SubwayActions.RIGHT: 0.35,
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

        self.vector_stack = deque(maxlen=4)
        for _ in range(4):
            self.vector_stack.append(np.zeros(8, dtype=np.float32))

        # RL Spaces
        self.action_space = spaces.Discrete(5)

        # 8 features * 4 historical frames = 32 flat values
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(32,),
            dtype=np.float32,
        )

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _get_stacked_observation(self):

        return np.concatenate(list(self.vector_stack)).flatten()

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

        if self.profile_steps != PROFILE_INTERVAL:
            return

        if PROFILE_ENV:
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

        stacked = self._get_stacked_observation()

        self.profile["process"] += time.perf_counter() - t
        self.profile["total"] += time.perf_counter() - step_start

        if DEBUG_OBSERVATION:

            annotated_frame = self.tracker.last_results[0].plot() if hasattr(self.tracker, 'last_results') and self.tracker.last_results else frame
            self.live_debugger.show(
                frame=frame,
                crop=annotated_frame,
                stacked=stacked,
                action=action,
                reward=reward,
                state=state,
                episode_steps=self.episode_steps,
                env_sps=(
                    (self.profile_steps + 1) / self.profile["total"]
                    if self.profile["total"] > 0
                    else 0
                ),
                reward_breakdown=reward_breakdown,
                detector_debug=self.detector.last_debug,
                on_train=self.on_train,
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
                0.05,
            ),
            "action_penalty": 0.0,
            "death": 0.0,
            "terminal_action_penalty": 0.0,
        }

        return breakdown

    def _terminal_action_penalty(self):

        penalty = 0.0
        
        # Weights for the last 5 actions (oldest to newest)
        weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        
        # Align weights with the actual number of actions in history
        active_weights = weights[-len(self.action_history):] if self.action_history else []

        breakdown_strs = []
            
        for i, action in enumerate(self.action_history):
            action_names_map = {0: "LEFT", 1: "RIGHT", 2: "JUMP", 3: "ROLL", 4: "IDLE"}
            action_name = action_names_map.get(int(action), str(action))
            step_pen = 0.0
            
            step_pen = 0.0 # ALL ACTION PENALTIES REMOVED per user request
            penalty += step_pen
            break_str = f"t-{len(self.action_history) - i - 1}: {action_name}({step_pen:+.2f})"
            breakdown_strs.append(break_str)

        return penalty, breakdown_strs

    def _game_over_reward(self, action, include_action_history_penalty):

        reward, _ = self.reward_system.compute(
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
        self.vector_stack.clear()
        for _ in range(4):
            self.vector_stack.append(np.zeros(8, dtype=np.float32))
        self.current_agent_lane = 1
        self.action_history.clear()
        self.reward_system.reset()
        self.episode_steps = 0
        self.detector.reset_episode()
        self.last_action_times.clear()
        self.on_train = False
        self.on_train_steps = 0

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
        self.vector_stack.clear()
        # Grab initial frames to fill the stack
        for _ in range(4):
            time.sleep(STEP_DELAY)
            frame = self.capture.grab()
            tracked_entities = self.tracker.track_frame(frame)
            vector = self.vector_builder.build_vector(tracked_entities, self.current_agent_lane, frame.shape, self.tracker.model.names)
            self.vector_stack.append(vector)

        # Accurately set the initial police state based on the first frames
        self.prev_police_present = 1.0 if self.vector_stack[-1][-1] == 1.0 else 0.0

        return self._get_stacked_observation(), {}

    def step(self, action):
        self.total_env_steps += 1
            
        step_start = time.perf_counter()
        now = time.perf_counter()

        # Check if agent is currently locked in an animation
        action_active = False
        for act, last_t in self.last_action_times.items():
            if act != SubwayActions.IDLE:
                if now - last_t < self.action_hold.get(act, 0.0):
                    action_active = True
                    break

        if action != SubwayActions.IDLE:
            self.last_action_times[action] = now

        # Update Lane Tracking
        previous_agent_lane = self.current_agent_lane
        if action == SubwayActions.LEFT:
            self.current_agent_lane = max(0, self.current_agent_lane - 1)
        elif action == SubwayActions.RIGHT:
            self.current_agent_lane = min(2, self.current_agent_lane + 1)

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

        police_present = 0.0

        # Track entities
        t = time.perf_counter()
        tracked_entities = self.tracker.track_frame(frame)

        # Check if we climbed a ramp train (climber) in our lane
        y_max = frame.shape[0]
        for ent in tracked_entities:
            cls_name = self.tracker.model.names[ent["class"]]
            if cls_name == "climber":
                obj_lane = self.lane_detector.get_object_lane(ent["center_x"], ent["center_y"], self.current_agent_lane)
                if obj_lane == self.current_agent_lane:
                    norm_dist = max(0.0, min(1.0, (y_max - ent["center_y"]) / y_max))
                    if norm_dist <= 0.35:
                        self.on_train = True
                        self.on_train_steps = 25  # 2.5 seconds of roof time

        # Process frame and append to observation history using the correct lane perspective
        vector = self.vector_builder.build_vector(tracked_entities, self.current_agent_lane, frame.shape, self.tracker.model.names)
        self.vector_stack.append(vector)
        self.profile["process"] += time.perf_counter() - t

        # Determine terminal state and rewards
        if state == "USE_KEYS":
            t = time.perf_counter()
            reward, breakdown = self._game_over_reward(
                action=action,
                include_action_history_penalty=True,
            )
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
            
            # Distance is at index 1 + agent_lane (Left=1, Center=2, Right=3)
            current_lane_dist = vector[1 + self.current_agent_lane]
            # Type is at index 4 + agent_lane (Left=4, Center=5, Right=6)
            current_lane_type = vector[4 + self.current_agent_lane]
            
            # Get data from the frame BEFORE the action was taken to see what the agent was reacting to
            previous_vector = self.vector_stack[-2] # -1 is the new frame, -2 is the old frame
            previous_lane_dist = previous_vector[1 + previous_agent_lane]
            previous_lane_type = previous_vector[4 + previous_agent_lane]
            
            # Previous action for spam cooldown
            previous_action = self.action_history[-1] if self.action_history else None

            reward, action_breakdown = self.reward_system.compute(
                state=state,
                action=action,
                episode_steps=self.episode_steps,
                police_present=False,
                stumbled=False,
                agent_lane=previous_agent_lane,
                new_agent_lane=self.current_agent_lane,
                current_lane_distance=current_lane_dist,
                current_lane_type=current_lane_type,
                previous_lane_distance=previous_lane_dist,
                previous_lane_type=previous_lane_type,
                action_active=action_active,
                on_train=self.on_train,
                previous_action=previous_action,
            )
            breakdown = self._running_reward_breakdown(action)
            # Merge breakdown
            breakdown.update(action_breakdown)
            
            self.profile["reward"] += time.perf_counter() - t

            # Decay climber/on_train status
            if self.on_train:
                self.on_train_steps -= 1
                if self.on_train_steps <= 0:
                    self.on_train = False

            if action in (SubwayActions.LEFT, SubwayActions.RIGHT):
                self.on_train = False
                self.on_train_steps = 0

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
