import cv2
import numpy as np
from collections import deque


class LiveObservationDebugger:

    # Layout constants
    CROP_W    = 400
    STACK_W   = 400
    WINDOW_W  = CROP_W + STACK_W   # 800
    INFO_H    = 180
    CONTENT_H = 520
    WINDOW_H  = CONTENT_H + INFO_H  # 700

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

    def _crop_panel(self, crop):
        """Left panel: gameplay crop displayed in color."""

        img = (
            crop.copy()
            if len(crop.shape) == 3
            else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        )

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

    def _stack_panel(self, stacked):
        """Right panel: 2x2 grid of the last 4 PPO input frames."""

        cw = self.STACK_W  // 2   # 200
        ch = self.CONTENT_H // 2  # 260

        labels = ["t-3", "t-2", "t-1", "NOW"]
        cells  = []

        for i in range(4):

            if stacked is not None and i < stacked.shape[2]:
                raw = stacked[:, :, i]
            else:
                raw = np.zeros((64, 64), dtype=np.uint8)

            cell = cv2.cvtColor(
                cv2.resize(raw, (cw - 2, ch - 2), interpolation=cv2.INTER_NEAREST),
                cv2.COLOR_GRAY2BGR,
            )

            label_color = self.GREEN if i == 3 else self.GRAY
            cv2.putText(cell, labels[i], (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 1)

            border = self.GREEN if i == 3 else (40, 40, 40)
            cell   = cv2.copyMakeBorder(cell, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=border)
            cells.append(cell)

        panel = np.vstack([
            np.hstack([cells[0], cells[1]]),
            np.hstack([cells[2], cells[3]]),
        ])

        cv2.putText(panel, "PPO FRAME STACK", (10, self.CONTENT_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.CYAN, 1)

        return panel

    # -------------------------------------------------------

    def _info_bar(self, state, action_name, reward, episode_steps, env_sps, reward_breakdown, detector_debug=None):
        """Bottom info bar with multi-row layout to prevent overflow."""

        bar  = np.full((self.INFO_H, self.WINDOW_W, 3), (18, 22, 18), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        sc   = 0.45

        cv2.line(bar, (0, 0), (self.WINDOW_W, 0), self.CYAN, 1)

        state_color  = self.GREEN if state == "RUNNING" else self.RED
        action_color = self.ACTION_COLORS.get(action_name, self.WHITE)

        # Row 1: State & Action
        x = 12
        for text, color in [
            ("STATE: ",          self.GRAY),
            (state,              state_color),
            ("      ACTION: ",   self.GRAY),
            (action_name,        action_color),
            (f"      STEPS: {episode_steps}", self.WHITE),
            (f"      BEST: {self.best_episode}", self.GRAY),
        ]:
            cv2.putText(bar, text, (x, 22), font, sc, color, 1)
            (tw, _), _ = cv2.getTextSize(text, font, sc, 1)
            x += tw

        # Row 2: Performance & Episode Reward
        x = 12
        for text, color in [
            (f"SPS: {env_sps:.1f}   ",            self.WHITE),
            (f"FPS: {self.current_fps:.1f}   ",   self.WHITE),
            (f"STEP REWARD: {reward:+.3f}   ",    self.YELLOW),
            (f"EP REWARD: {self.total_reward:.2f}", self.YELLOW),
        ]:
            cv2.putText(bar, text, (x, 44), font, sc, color, 1)
            (tw, _), _ = cv2.getTextSize(text, font, sc, 1)
            x += tw

        # Row 3: Reward Breakdown
        x = 12
        if reward_breakdown:
            cv2.putText(bar, "BREAKDOWN: ", (x, 66), font, sc, self.GRAY, 1)
            (tw, _), _ = cv2.getTextSize("BREAKDOWN: ", font, sc, 1)
            x += tw
            for k, v in reward_breakdown.items():
                if k == "terminal_strings":
                    continue
                if abs(v) > 1e-6:
                    text  = f"{k}: {v:+.3f}   "
                    color = self.GREEN if v > 0 else self.RED
                    cv2.putText(bar, text, (x, 66), font, sc, color, 1)
                    (tw, _), _ = cv2.getTextSize(text, font, sc, 1)
                    x += tw

        # Row 4: Detectors
        x = 12
        cv2.putText(bar, "DETECTORS: ", (12, 88), font, sc, self.GRAY, 1)
        (tw, _), _ = cv2.getTextSize("DETECTORS: ", font, sc, 1)
        x += tw
        if detector_debug:
            total_ms = 0.0
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
                    text = f"{det_name.upper()}: {status_str}   "
                    cv2.putText(bar, text, (x, 88), font, 0.42, color, 1)
                    (tw, _), _ = cv2.getTextSize(text, font, 0.42, 1)
                    x += tw
            cv2.putText(bar, f"TOTAL: {total_ms:.1f}ms", (self.WINDOW_W - 140, 88), font, sc, self.CYAN, 1)

        # Row 5 & 6: Action History
        hist_list = list(self.action_history)
        history_1 = "  \u2192  ".join(hist_list[:8])
        cv2.putText(bar, f"HISTORY: {history_1}", (12, 110), font, 0.42, (170, 210, 170), 1)

        history_2 = "  \u2192  ".join(hist_list[8:]) if len(hist_list) > 8 else ""
        if history_2:
            cv2.putText(bar, f"         {history_2}", (12, 132), font, 0.42, (170, 210, 170), 1)

        # Row 7: Terminal Penalty Breakdown (Only visible if dead)
        if reward_breakdown and "terminal_strings" in reward_breakdown:
            penalty_str = " | ".join(reward_breakdown["terminal_strings"])
            cv2.putText(bar, f"DEATH PENALTY DISTRIBUTION: {penalty_str}", (12, 154), font, 0.42, self.RED, 1)

        cv2.line(bar, (0, self.INFO_H - 1), (self.WINDOW_W, self.INFO_H - 1), self.CYAN, 1)

        return bar

    # -------------------------------------------------------
    # Main entry — called once per completed PPO action
    # -------------------------------------------------------

    def show(
        self,
        frame,
        processor,
        crop,
        gray,
        stacked,
        action,
        reward,
        state,
        episode_steps,
        env_sps=0.0,
        reward_breakdown=None,
        detector_debug=None,
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

        # Build panels
        crop_panel  = self._crop_panel(crop)
        stack_panel = self._stack_panel(stacked)
        info_bar    = self._info_bar(
            state, action_name, reward,
            episode_steps, env_sps, reward_breakdown,
            detector_debug=detector_debug,
        )

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
