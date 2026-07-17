import cv2
import numpy as np
from collections import deque
from vision.lane_detector import LaneDetector


class LiveObservationDebugger:

    # Layout constants
    CROP_W    = 400
    STACK_W   = 600
    WINDOW_W  = CROP_W + STACK_W   # 1000
    INFO_H    = 60
    CONTENT_H = 770
    WINDOW_H  = CONTENT_H + INFO_H  # 830

    # BGR colors
    BG        = (15, 15, 15)
    PANEL_BG  = (22, 28, 22)
    GREEN     = (80, 255, 60)
    RED       = (60, 60, 240)
    CYAN      = (210, 190, 0)
    WHITE     = (245, 245, 245)
    GRAY      = (155, 155, 155)
    YELLOW    = (0, 210, 210)

    ACTION_COLORS = {
        "IDLE":  (140, 140, 140),
        "LEFT":  (0, 190, 255),
        "RIGHT": (0, 190, 255),
        "JUMP":  (80, 255, 60),
        "ROLL":  (200, 90, 255),
    }

    ACTION_NAMES = {0: "LEFT", 1: "RIGHT", 2: "JUMP", 3: "ROLL", 4: "IDLE"}

    # -------------------------------------------------------

    def __init__(self):

        self.window = "RL Vision Studio"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, self.WINDOW_W, self.WINDOW_H)
        cv2.moveWindow(self.window, 40, 40)

        self.action_history = deque(maxlen=16)
        self.total_reward   = 0.0
        self.best_episode   = 0
        self.frame_counter  = 0
        self.last_time      = cv2.getTickCount()
        self.current_fps    = 0.0
        
        self.lane_detector  = LaneDetector()

    # -------------------------------------------------------

    def _update_fps(self):

        self.frame_counter += 1

        if self.frame_counter < 5:
            return

        now = cv2.getTickCount()
        dt  = (now - self.last_time) / cv2.getTickFrequency()

        if dt > 0:
            self.current_fps = self.frame_counter / dt

        self.frame_counter = 0
        self.last_time     = now

    # -------------------------------------------------------

    def _crop_panel(self, crop, agent_lane_val):
        """Left panel: gameplay crop displayed in color."""

        img = (
            crop.copy()
            if len(crop.shape) == 3
            else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        )
        
        # Determine int lane based on vector value (0.0=0, 0.5=1, 1.0=2)
        if agent_lane_val < 0.25:
            agent_lane = 0
        elif agent_lane_val > 0.75:
            agent_lane = 2
        else:
            agent_lane = 1
            
        # Draw dynamic perspective lane boundaries using the mathematically perfect detector
        img = self.lane_detector.draw(img, agent_lane)

        h, w   = img.shape[:2]
        scale  = min(self.CROP_W / w, self.CONTENT_H / h)
        nw, nh = int(w * scale), int(h * scale)
        img    = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        panel = np.full((self.CONTENT_H, self.CROP_W, 3), self.PANEL_BG, dtype=np.uint8)
        y0 = (self.CONTENT_H - nh) // 2
        x0 = (self.CROP_W - nw) // 2
        panel[y0:y0 + nh, x0:x0 + nw] = img

        cv2.putText(panel, "GAMEPLAY CROP", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.CYAN, 1)
        cv2.rectangle(panel, (1, 1), (self.CROP_W - 2, self.CONTENT_H - 2), self.CYAN, 1)

        return panel

    # -------------------------------------------------------

    def _stack_panel(self, stacked, state, action_name, reward, episode_steps, env_sps, reward_breakdown, detector_debug, on_train):
        """Right panel: Renders the latest state vector and all metadata."""
        
        panel = np.full((self.CONTENT_H, self.STACK_W, 3), self.PANEL_BG, dtype=np.uint8)
        cv2.putText(panel, "VECTOR STATE SPACE", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.CYAN, 1)
        cv2.rectangle(panel, (1, 1), (self.STACK_W - 2, self.CONTENT_H - 2), self.CYAN, 1)

        if stacked is None or len(stacked.shape) == 0:
            return panel
            
        # The vector is (32,) consisting of 4 frames of 8 features.
        # We'll display the latest frame (the last 8 features)
        latest_frame = stacked[-8:] if len(stacked.shape) == 1 else stacked[0, -8:]
        
        agent_lane = latest_frame[0]
        distances = latest_frame[1:4]
        types = latest_frame[4:7]
        police_dist = latest_frame[7]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 1. Agent Lane
        lane_str = "LEFT" if agent_lane < 0.25 else "RIGHT" if agent_lane > 0.75 else "CENTER"
        cv2.putText(panel, f"Agent Lane: {lane_str} ({agent_lane:.2f})", (15, 60), font, 0.6, self.WHITE, 1)
        
        # 2. Obstacles
        cv2.putText(panel, "Obstacle Distances & Types:", (15, 100), font, 0.6, self.CYAN, 1)
        
        lanes = ["Left", "Center", "Right"]
        for i in range(3):
            y = 130 + (i * 40)
            dist = distances[i]
            obj_type = types[i]
            
            # Map type back to string for UI
            if obj_type == 0.0:
                t_str = "Clear"
                t_color = self.GREEN
            elif obj_type == 0.2:
                t_str = "[Climb / Idle]"
                t_color = self.GREEN
            elif obj_type == 0.4:
                t_str = "[JUMP Required]"
                t_color = (255, 100, 255) # Pink
            elif obj_type == 0.6:
                t_str = "[ROLL Required]"
                t_color = (0, 165, 255) # Orange
            elif obj_type == 1.0:
                t_str = "[LANE CHANGE]"
                t_color = (50, 50, 255) # Red
            else:
                t_str = f"Unknown ({obj_type:.2f})"
                t_color = self.GRAY
                
            # Draw text
            text = f"{lanes[i]}: {dist:.2f} [{t_str}]"
            cv2.putText(panel, text, (15, y), font, 0.5, t_color, 1)
            
            # Draw bar
            bar_w = 150
            bar_x = 220
            # Distance 1.0 = clear (far), 0.0 = close
            fill_w = int(dist * bar_w)
            # Red when close, Green when far
            bar_color = (0, 0, 255) if dist < 0.3 else (0, 255, 255) if dist < 0.6 else (0, 255, 0)
            
            cv2.rectangle(panel, (bar_x, y - 12), (bar_x + bar_w, y + 2), (50, 50, 50), -1)
            cv2.rectangle(panel, (bar_x, y - 12), (bar_x + fill_w, y + 2), bar_color, -1)
            
        # 3. Police Present & Climbed
        cv2.putText(panel, "Police Present:", (15, 270), font, 0.6, self.CYAN, 1)
        p_str = "YES" if police_dist == 1.0 else "NO"
        p_color = self.RED if police_dist == 1.0 else self.GREEN
        cv2.putText(panel, p_str, (15, 300), font, 0.6, p_color, 1)
        
        cv2.putText(panel, "Climbed (On Train):", (250, 270), font, 0.6, self.CYAN, 1)
        c_str = "YES" if on_train else "NO"
        c_color = self.GREEN if on_train else self.GRAY
        cv2.putText(panel, c_str, (250, 300), font, 0.6, c_color, 1)
        
        # -----------------------------------------------------------
        # 4. METADATA (Moved from bottom bar)
        # -----------------------------------------------------------
        sc = 0.5
        y = 350
        
        state_color  = self.GREEN if state == "RUNNING" else self.RED
        action_color = self.ACTION_COLORS.get(action_name, self.WHITE)

        # Row 1: State & Action
        cv2.putText(panel, f"STATE: {state}", (15, y), font, sc, state_color, 1)
        cv2.putText(panel, f"ACTION: {action_name}", (250, y), font, sc, action_color, 1)
        
        y += 35
        # Row 2: Steps
        cv2.putText(panel, f"STEPS: {episode_steps}", (15, y), font, sc, self.WHITE, 1)
        cv2.putText(panel, f"BEST EPISODE: {self.best_episode}", (250, y), font, sc, self.GRAY, 1)

        y += 35
        # Row 3: Performance
        cv2.putText(panel, f"SPS: {env_sps:.1f}", (15, y), font, sc, self.WHITE, 1)
        cv2.putText(panel, f"FPS: {self.current_fps:.1f}", (250, y), font, sc, self.WHITE, 1)

        y += 35
        # Row 4: Rewards
        cv2.putText(panel, f"STEP REWARD: {reward:+.3f}", (15, y), font, sc, self.YELLOW, 1)
        cv2.putText(panel, f"EPISODE REWARD: {self.total_reward:.2f}", (250, y), font, sc, self.YELLOW, 1)

        y += 45
        # Row 5: Reward Breakdown
        if reward_breakdown:
            cv2.putText(panel, "REWARD BREAKDOWN:", (15, y), font, sc, self.CYAN, 1)
            y += 25
            
            x_offset = 15
            for k, v in reward_breakdown.items():
                if k == "terminal_strings":
                    continue
                if abs(v) > 1e-6:
                    text  = f"{k}: {v:+.3f} "
                    color = self.GREEN if v > 0 else self.RED
                    cv2.putText(panel, text, (x_offset, y), font, 0.45, color, 1)
                    (tw, _), _ = cv2.getTextSize(text, font, 0.45, 1)
                    x_offset += tw + 15
                    if x_offset > self.STACK_W - 150:
                        x_offset = 15
                        y += 25

        y += 45
        # Row 6: Detectors
        cv2.putText(panel, "DETECTORS:", (15, y), font, sc, self.CYAN, 1)
        y += 25
        if detector_debug:
            total_ms = 0.0
            x_offset = 15
            for det_name, data in detector_debug.items():
                ms = data.get("detect_ms", 0.0)
                matched = data.get("matched", False)
                if ms > 0.0 or matched:
                    total_ms += ms
                    if det_name == "pause_button":
                        color = self.GREEN if matched else self.RED
                        status_str = "MATCH" if matched else f"MISSING ({ms:.1f}ms)"
                    else:
                        color = self.RED if matched else self.GREEN
                        status_str = "MATCH" if matched else f"{ms:.1f}ms"
                        
                    text = f"{det_name.upper()}: {status_str}"
                    cv2.putText(panel, text, (x_offset, y), font, 0.42, color, 1)
                    (tw, _), _ = cv2.getTextSize(text, font, 0.42, 1)
                    x_offset += tw + 15
                    if x_offset > self.STACK_W - 150:
                        x_offset = 15
                        y += 25
                        
            y += 25
            cv2.putText(panel, f"TOTAL DETECT TIME: {total_ms:.1f}ms", (15, y), font, sc, self.CYAN, 1)

        return panel

    # -------------------------------------------------------

    def _info_bar(self, reward_breakdown):
        """Bottom info bar dedicated to history to prevent overflow."""

        bar  = np.full((self.INFO_H, self.WINDOW_W, 3), (18, 22, 18), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        sc   = 0.45

        cv2.line(bar, (0, 0), (self.WINDOW_W, 0), self.CYAN, 1)

        # Action History
        hist_list = list(self.action_history)
        history_1 = "  \u2192  ".join(hist_list[:8])
        cv2.putText(bar, f"HISTORY: {history_1}", (12, 22), font, 0.45, (170, 210, 170), 1)

        history_2 = "  \u2192  ".join(hist_list[8:]) if len(hist_list) > 8 else ""
        if history_2:
            cv2.putText(bar, f"         {history_2}", (12, 44), font, 0.45, (170, 210, 170), 1)

        # Terminal Penalty Breakdown (Only visible if dead)
        if reward_breakdown and "terminal_strings" in reward_breakdown:
            penalty_str = " | ".join(reward_breakdown["terminal_strings"])
            cv2.putText(bar, f"DEATH PENALTY: {penalty_str}", (12, 44 if not history_2 else 60), font, 0.42, self.RED, 1)

        return bar

    # -------------------------------------------------------
    # Main entry — called once per completed PPO action
    # -------------------------------------------------------

    def show(
        self,
        frame,
        crop,
        stacked,
        action,
        reward,
        state,
        episode_steps,
        env_sps=0.0,
        reward_breakdown=None,
        detector_debug=None,
        on_train=False,
    ):
        self._update_fps()

        action_name = self.ACTION_NAMES.get(int(action), str(action))

        # Episode stats
        if state == "RUNNING":
            self.total_reward += reward
        elif state == "GAME_OVER":
            self.best_episode = max(self.best_episode, episode_steps)
            self.total_reward = 0.0

        # Action history — Log every single PPO step exactly, along with its reward
        history_str = f"{action_name} ({reward:+.2f})"
        self.action_history.append(history_str)

        # Extract agent lane for dynamic boundary drawing
        agent_lane_val = 0.5
        if stacked is not None and len(stacked.shape) > 0:
            latest_frame = stacked[-8:] if len(stacked.shape) == 1 else stacked[0, -8:]
            agent_lane_val = latest_frame[0]

        # Build panels
        crop_panel  = self._crop_panel(crop, agent_lane_val)
        stack_panel = self._stack_panel(
            stacked, state, action_name, reward, episode_steps, 
            env_sps, reward_breakdown, detector_debug, on_train
        )
        info_bar    = self._info_bar(reward_breakdown)

        content   = np.hstack((crop_panel, stack_panel))
        dashboard = np.vstack((content, info_bar))

        # Game over flash
        if state == "GAME_OVER":
            overlay = dashboard.copy()
            cv2.rectangle(overlay, (0, 0), (self.WINDOW_W, 55), (25, 25, 160), -1)
            cv2.addWeighted(overlay, 0.55, dashboard, 0.45, 0, dashboard)
            cv2.putText(
                dashboard, "GAME OVER",
                (self.WINDOW_W // 2 - 125, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.WHITE, 3,
            )

        cv2.imshow(self.window, dashboard)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            cv2.imwrite("debug_crop.png", crop)
            cv2.imwrite("debug_frame.png", frame)
            print("Saved debug images.")
        elif key == ord("q"):
            cv2.destroyWindow(self.window)

    # -------------------------------------------------------

    def close(self):

        try:
            cv2.destroyWindow(self.window)
        except Exception:
            pass
