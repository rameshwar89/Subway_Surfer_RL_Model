import json
import time
import cv2
import numpy as np

class StateDetector:

    def __init__(self):

        with open("assets/configs/calibration.json") as f:
            full_config = json.load(f)
            
        detection_config = full_config["state_detection"]

        self.last_debug = {}
        self.pause_missing_frames = 0
        
        roi_cfg = detection_config["rois"]["pause_button"]
        self.roi = (roi_cfg["x"], roi_cfg["y"], roi_cfg["width"], roi_cfg["height"])
        self.threshold = detection_config["thresholds"]["pause_button"]
        
        template_path = "assets/patches/ui/pause_button/pause_button_active.png"
        self.template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if self.template is None:
            raise FileNotFoundError(f"Could not load pause button template at {template_path}")

        print("\n========== DETECTOR HEALTH ==========")
        print(f"Pause Button detector initialized (Standalone)")
        print("=====================================\n")

    def reset_episode(self):
        self.last_debug = {}
        self.pause_missing_frames = 0

    def detect(self, frame, context="all"):
        t0 = time.perf_counter()
        
        # Crop the ROI
        x, y, w, h = self.roi
        crop = frame[y:y+h, x:x+w]
        
        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Match template
        res = cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        matched = max_val >= self.threshold
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        state_key = "pause_button"
        
        self.last_debug = {
            state_key: {
                "score": float(max_val),
                "matched": matched,
                "votes": 1 if matched else 0,
                "detect_ms": elapsed_ms,
            }
        }
        
        all_scores = {"PAUSE_BUTTON": float(max_val)}
        
        if not matched:
            self.pause_missing_frames += 1
            if self.pause_missing_frames >= 1:
                return "GAME_OVER_UI", 0, all_scores
        else:
            self.pause_missing_frames = 0
            
        return "RUNNING", 0, all_scores
