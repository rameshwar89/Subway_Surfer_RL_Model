import cv2
import numpy as np
from collections import deque
from vision.lane_detector import LaneDetector


class LiveObservationDebugger:

    # Layout constants
    CROP_W    = 390
    STACK_W   = 610
    WINDOW_W  = CROP_W + STACK_W   # 1000
    INFO_H    = 50
    CONTENT_H = 770
    WINDOW_H  = CONTENT_H + INFO_H  # 820

    # BGR colors — terminal-green aesthetic
    PANEL_BG  = (14, 18, 14)
    BORDER    = (0, 160, 80)
    GREEN     = (60, 220, 60)
    RED       = (50, 50, 220)
    ORANGE    = (0, 140, 255)
    CYAN      = (180, 210, 0)
    WHITE     = (230, 230, 230)
    GRAY      = (110, 110, 110)
    YELLOW    = (0, 200, 200)
    DIM       = (70, 90, 70)

    # Section divider color
    DIVIDER   = (0, 60, 30)

    ACTION_COLORS = {
        "IDLE":  (120, 120, 120),
        "LEFT":  (30, 180, 255),
        "RIGHT": (30, 180, 255),
        "JUMP":  (60, 220, 60),
        "ROLL":  (200, 80, 255),
    }

    ACTION_NAMES = {0: "LEFT", 1: "RIGHT", 2: "JUMP", 3: "ROLL", 4: "IDLE"}

    # Obstacle type encoding -> (label, color)
    TYPE_MAP = {
        0.0: ("CLEAR",       (60, 220, 60)),
        0.2: ("CLIMBER",     (60, 220, 60)),
        0.3: ("BLOCKER",     (60, 180, 255)),   # Jump OR Roll
        0.4: ("ROLL-BLOCK",  (200, 80, 255)),   # Jump only
        0.6: ("JUMP-BAR",    (0, 140, 255)),    # Roll only
        1.0: ("TRAIN",       (50, 50, 220)),    # Lane change
    }

    # -------------------------------------------------------

    def __init__(self, initial_episode=0):

        self.window = "RL Vision Studio"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, self.WINDOW_W, self.WINDOW_H)
        cv2.moveWindow(self.window, 40, 40)

        # State tracking
        self.action_history  = deque(maxlen=12)
        self.total_reward    = 0.0
        self.best_reward     = -1e9
        self.best_steps      = 0
        self.episode_num     = initial_episode

        # Session-level stats for action distribution and episode trend
        self.session_actions = {"LEFT": 0, "RIGHT": 0, "JUMP": 0, "ROLL": 0, "IDLE": 0}
        self.episode_history = deque(maxlen=10)  # (steps, reward) per episode

        # FPS tracking (update every N frames)
        self.frame_counter   = 0
        self.last_tick       = cv2.getTickCount()
        self.current_fps     = 0.0

        self.lane_detector   = LaneDetector()

        # Pre-allocate panel buffers to avoid allocating every frame
        self._crop_buf  = np.full((self.CONTENT_H, self.CROP_W, 3),  self.PANEL_BG, dtype=np.uint8)
        self._stack_buf = np.full((self.CONTENT_H, self.STACK_W, 3), self.PANEL_BG, dtype=np.uint8)
        self._bar_buf   = np.full((self.INFO_H,    self.WINDOW_W, 3), self.PANEL_BG, dtype=np.uint8)

    # -------------------------------------------------------

    def _update_fps(self):
        self.frame_counter += 1
        if self.frame_counter < 6:
            return
        now = cv2.getTickCount()
        dt  = (now - self.last_tick) / cv2.getTickFrequency()
        if dt > 0:
            self.current_fps = self.frame_counter / dt
        self.frame_counter = 0
        self.last_tick     = now

    # -------------------------------------------------------

    def _label(self, img, text, x, y, scale=0.45, color=None, bold=False):
        """Tiny helper to draw text with one call."""
        color = color or self.WHITE
        thickness = 2 if bold else 1
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def _hline(self, img, y, w=None):
        w = w or img.shape[1]
        cv2.line(img, (8, y), (w - 8, y), self.DIVIDER, 1)

    def _bar_rect(self, img, x, y, w, h, fill, bg=(40, 40, 40)):
        """Draw a small progress bar."""
        cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
        fw = max(0, min(w, int(fill * w)))
        if fw > 0:
            cv2.rectangle(img, (x, y), (x + fw, y + h), self._dist_color(fill), -1)

    def _dist_color(self, dist):
        """Red when close (0), yellow mid (0.5), green far (1.0)."""
        if dist < 0.35:
            return (30, 30, 200)   # red
        if dist < 0.60:
            return (0, 200, 200)   # yellow
        return (40, 200, 40)       # green

    # -------------------------------------------------------
    # LEFT PANEL — gameplay feed with lane overlay
    # -------------------------------------------------------

    def _crop_panel(self, crop, agent_lane_int):
        """Resize and annotate the raw gameplay feed."""

        img = crop.copy() if len(crop.shape) == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        img = self.lane_detector.draw(img, agent_lane_int)

        h, w  = img.shape[:2]
        scale = min(self.CROP_W / w, self.CONTENT_H / h)
        nw    = int(w * scale)
        nh    = int(h * scale)
        img   = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        panel = self._crop_buf
        panel[:] = self.PANEL_BG
        y0 = (self.CONTENT_H - nh) // 2
        x0 = (self.CROP_W    - nw) // 2
        panel[y0:y0 + nh, x0:x0 + nw] = img

        cv2.rectangle(panel, (1, 1), (self.CROP_W - 2, self.CONTENT_H - 2), self.BORDER, 1)
        self._label(panel, "GAME FEED", 8, 18, 0.42, self.BORDER)
        return panel

    # -------------------------------------------------------
    # RIGHT PANEL — state vector + metadata
    # -------------------------------------------------------

    def _stack_panel(self, stacked, state, action_name, reward,
                     episode_steps, env_sps, reward_breakdown,
                     detector_debug, on_train):

        W  = self.STACK_W
        panel = self._stack_buf
        panel[:] = self.PANEL_BG
        cv2.rectangle(panel, (1, 1), (W - 2, self.CONTENT_H - 2), self.BORDER, 1)

        if stacked is None or stacked.size == 0:
            return panel

        # Decode state vector — latest frame is last 8 floats of (32,) flat array
        vec         = stacked[-8:]
        agent_lane  = vec[0]           # 0.0=L, 0.5=C, 1.0=R
        distances   = vec[1:4]         # normalized 0-1, 1.0=far/clear
        obs_types   = vec[4:7]         # semantic type codes
        police      = vec[7]           # 1.0 = police present

        lane_int = 0 if agent_lane < 0.25 else (2 if agent_lane > 0.75 else 1)
        lane_str = ("LEFT", "CENTER", "RIGHT")[lane_int]

        # ── Uniform type sizes ────────────────────────────────────
        # All data text:    scale=0.48
        # Section headers:  scale=0.46 bold + BORDER color
        # Pill text:        scale=0.45 bold, dark bg
        S  = 0.48   # standard data size
        SH = 0.46   # section header size

        y = 22

        # ── SECTION: AGENT STATUS ────────────────────────────────
        self._label(panel, "AGENT STATUS", 10, y, SH, self.BORDER, bold=True)
        y += 24

        # Lane pill
        lane_color = (30, 180, 255) if lane_int != 1 else self.GREEN
        cv2.rectangle(panel, (10, y - 14), (100, y + 4), lane_color, -1)
        self._label(panel, lane_str, 14, y, 0.45, (10, 10, 10), bold=True)

        # Action pill
        a_col = self.ACTION_COLORS.get(action_name, self.WHITE)
        cv2.rectangle(panel, (110, y - 14), (215, y + 4), a_col, -1)
        self._label(panel, action_name, 114, y, 0.45, (10, 10, 10), bold=True)

        # State pill
        s_col = self.GREEN if state == "RUNNING" else self.RED
        cv2.rectangle(panel, (225, y - 14), (380, y + 4), s_col, -1)
        self._label(panel, state, 229, y, 0.45, (10, 10, 10), bold=True)

        # On-train badge
        if on_train:
            cv2.rectangle(panel, (390, y - 14), (500, y + 4), self.ORANGE, -1)
            self._label(panel, "ON TRAIN", 394, y, 0.45, (10, 10, 10), bold=True)

        y += 28
        self._hline(panel, y)
        y += 14

        # ── SECTION: OBSTACLES ────────────────────────────────────
        self._label(panel, "OBSTACLES", 10, y, SH, self.BORDER, bold=True)
        y += 24

        # Fixed column grid
        LANE_X = 10    # lane letter
        DIST_X = 35    # distance value
        BAR_X  = 95    # bar start
        BAR_W  = 210   # bar width
        TYPE_X = 315   # type label

        for i in range(3):
            dist  = float(distances[i])
            otype = float(obs_types[i])
            t_label, t_col = self.TYPE_MAP.get(
                min(self.TYPE_MAP.keys(), key=lambda k: abs(k - otype)),
                (f"{otype:.1f}", self.GRAY)
            )
            is_agent = (i == lane_int)
            lbl_col  = self.YELLOW if is_agent else self.WHITE

            self._label(panel, ("L", "C", "R")[i], LANE_X, y, S, lbl_col, bold=is_agent)
            self._label(panel, f"{dist:.2f}",       DIST_X, y, S, self._dist_color(dist))
            self._bar_rect(panel, BAR_X, y - 13, BAR_W, 15, dist)
            self._label(panel, t_label,             TYPE_X, y, S, t_col)
            y += 26

        y += 2
        self._hline(panel, y)
        y += 14

        # ── SECTION: EPISODE STATS ────────────────────────────────
        self._label(panel, "EPISODE STATS", 10, y, SH, self.BORDER, bold=True)
        y += 24

        # Three-column row — uniform size
        C1, C2, C3 = 10, 175, 380
        self._label(panel, f"EP #{self.episode_num}",   C1, y, S, self.WHITE, bold=True)
        self._label(panel, f"STEPS   {episode_steps}",  C2, y, S, self.WHITE)
        self._label(panel, f"BEST   {self.best_steps}", C3, y, S, self.GRAY)
        y += 26

        if self.episode_history:
            n           = len(self.episode_history)
            mean_steps  = sum(e[0] for e in self.episode_history) / n
            mean_reward = sum(e[1] for e in self.episode_history) / n
            self._label(panel, f"AVG ({n} ep)",         C1, y, S, self.CYAN)
            self._label(panel, f"steps {mean_steps:.0f}", C2, y, S, self.WHITE)
            self._label(panel, f"rew {mean_reward:+.1f}", C3, y, S, self.YELLOW)
            y += 26

        self._label(panel, f"SPS {env_sps:.1f}",     C2,  y, S, self.WHITE)
        self._label(panel, f"FPS {self.current_fps:.1f}", C3, y, S, self.WHITE)
        y += 24

        self._hline(panel, y)
        y += 14

        # ── SECTION: ACTION DISTRIBUTION ─────────────────────────
        self._label(panel, "ACTION DISTRIBUTION", 10, y, SH, self.BORDER, bold=True)
        y += 24

        total_acts = max(1, sum(self.session_actions.values()))
        act_order  = [("LEFT",  (30, 180, 255)),
                      ("RIGHT", (30, 180, 255)),
                      ("JUMP",  (60, 220, 60)),
                      ("ROLL",  (200, 80, 255)),
                      ("IDLE",  (140, 140, 140))]

        ABAR_X = 90          # enough room for "RIGHT" at scale 0.48
        ABAR_W = W - ABAR_X - 65
        PCT_X  = ABAR_X + ABAR_W + 8

        for name, col in act_order:
            pct = self.session_actions[name] / total_acts
            self._label(panel, name, 10, y, S, col)
            cv2.rectangle(panel, (ABAR_X, y - 13), (ABAR_X + ABAR_W, y + 4), (35, 35, 35), -1)
            fw = max(0, int(pct * ABAR_W))
            if fw > 0:
                cv2.rectangle(panel, (ABAR_X, y - 13), (ABAR_X + fw, y + 4), col, -1)
            self._label(panel, f"{pct*100:.0f}%", PCT_X, y, S, self.WHITE)
            y += 22

        y += 6
        self._hline(panel, y)
        y += 14

        # ── SECTION: REWARDS ─────────────────────────────────────
        self._label(panel, "REWARDS", 10, y, SH, self.BORDER, bold=True)
        y += 24

        r_col = self.GREEN if reward > 0 else (self.RED if reward < -1 else self.YELLOW)
        self._label(panel, f"STEP  {reward:+.3f}",               C1,  y, S, r_col, bold=True)
        self._label(panel, f"EP TOTAL  {self.total_reward:+.2f}", C2,  y, S, self.YELLOW)
        self._label(panel, f"BEST  {self.best_reward:+.1f}",      C3,  y, S, self.GRAY)
        y += 26

        if reward_breakdown:
            items = [(k, v) for k, v in reward_breakdown.items()
                     if isinstance(v, (int, float)) and abs(v) > 1e-6]
            col_w = (W - 20) // 2
            cx    = [10, 10 + col_w]
            col   = 0
            for k, v in items:
                v_col   = self.GREEN if v > 0 else self.RED
                short_k = k.replace("_", " ")
                self._label(panel, f"{short_k}: {v:+.3f}", cx[col], y, S, v_col)
                col += 1
                if col > 1:
                    col = 0
                    y  += 22
            if col == 1:
                y += 22

        y += 6
        self._hline(panel, y)
        y += 14

        # ── SECTION: DETECTORS ───────────────────────────────────
        self._label(panel, "DETECTORS", 10, y, SH, self.BORDER, bold=True)
        y += 24

        if detector_debug:
            total_ms = 0.0
            for det_name, data in detector_debug.items():
                ms      = data.get("detect_ms", 0.0)
                matched = data.get("matched", False)
                total_ms += ms
                if det_name == "pause_button":
                    col    = self.GREEN if matched else self.RED
                    status = "OK" if matched else f"MISS {ms:.1f}ms"
                else:
                    col    = self.RED if matched else self.GREEN
                    status = "MATCH" if matched else f"{ms:.0f}ms"
                tag = f"{det_name.replace('_', ' ').upper()}: {status}"
                self._label(panel, tag, 10, y, S, col)
                y += 22

            self._label(panel, f"TOTAL DETECT TIME: {total_ms:.1f}ms", 10, y, S, self.CYAN)

        return panel

    # -------------------------------------------------------
    # BOTTOM BAR — action history ticker
    # -------------------------------------------------------

    def _info_bar(self):
        W   = self.WINDOW_W
        bar = self._bar_buf
        bar[:] = self.PANEL_BG
        cv2.line(bar, (0, 0), (W, 0), self.BORDER, 1)

        # Show last 10 actions, most recent on the right
        hist = list(self.action_history)[-10:]
        # Compact: "L(+0.20) -> R(-10.00)"
        items = []
        for h in hist:
            # strip the reward to save space if it's common survival
            items.append(h)
        text = "  |  ".join(items)
        self._label(bar, f"HISTORY  {text}", 10, 30, 0.40, self.DIM)

        return bar

    # -------------------------------------------------------
    # MAIN ENTRY
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

        # Track episode stats
        if state == "RUNNING":
            self.total_reward += reward
        elif state == "GAME_OVER":
            # Latch bests and log into rolling history before reset
            self.best_steps  = max(self.best_steps, episode_steps)
            ep_total = self.total_reward + reward
            self.best_reward = max(self.best_reward, ep_total)
            self.episode_history.append((episode_steps, ep_total))
            self.episode_num += 1
            self.total_reward = 0.0

        # Track session action distribution
        if action_name in self.session_actions:
            self.session_actions[action_name] += 1

        # Action history — compact label
        a_short = {"LEFT": "L", "RIGHT": "R", "JUMP": "J", "ROLL": "Ro", "IDLE": "I"}.get(action_name, action_name)
        history_str = f"{a_short}({reward:+.2f})"
        self.action_history.append(history_str)

        # Agent lane integer
        agent_lane_val = 0.5
        if stacked is not None and stacked.size > 0:
            agent_lane_val = float(stacked[-8])
        lane_int = 0 if agent_lane_val < 0.25 else (2 if agent_lane_val > 0.75 else 1)

        # Build panels
        crop_panel  = self._crop_panel(crop, lane_int)
        stack_panel = self._stack_panel(
            stacked, state, action_name, reward, episode_steps,
            env_sps, reward_breakdown, detector_debug, on_train
        )
        info_bar    = self._info_bar()

        content   = np.hstack((crop_panel, stack_panel))
        dashboard = np.vstack((content, info_bar))

        # Game Over flash overlay
        if state == "GAME_OVER":
            overlay = dashboard[:50, :].copy()
            cv2.rectangle(overlay, (0, 0), (self.WINDOW_W, 50), (20, 20, 140), -1)
            cv2.addWeighted(overlay, 0.65, dashboard[:50, :], 0.35, 0, dashboard[:50, :])
            self._label(dashboard, f"GAME OVER   EP #{self.episode_num}   STEPS {episode_steps}",
                        self.WINDOW_W // 2 - 220, 34, 0.85, self.WHITE, bold=True)

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
