from vision.menu_detector_modular import (
    GameOverUIDetector,
    LeaveConfirmDetector,
    MainMenuDetector,
)
from vision.patch_category_detector import load_calibration, PatchCategoryDetector
from vision.pause_menu_detector import PauseMenuDetector
from vision.popup_detector import PopupDetector


class StateDetector:

    def __init__(self):

        full_config = load_calibration()
        detection_config = full_config["state_detection"]

        self.last_debug = {}

        self.pause_missing_frames = 0

        self.use_keys_detector = PopupDetector(detection_config)
        self.game_over_detector = GameOverUIDetector(detection_config)
        self.leave_confirm_detector = LeaveConfirmDetector(
            detection_config,
            full_config,
        )
        self.pause_menu_detector = PauseMenuDetector(
            detection_config,
            full_config,
        )
        self.main_menu_detector = MainMenuDetector(detection_config)
        
        # Load the pause button patch dynamically as a category detector
        self.pause_button_detector = PatchCategoryDetector(
            patch_dir="assets/patches/ui/pause_button",
            roi=detection_config["rois"]["pause_button"],
            threshold=detection_config["thresholds"]["pause_button"],
            min_votes=detection_config["min_votes"]["pause_button"],
        )

        self.detectors = [
            ("PAUSE_BUTTON", self.pause_button_detector),
            ("USE_KEYS", self.use_keys_detector),
            ("GAME_OVER_UI", self.game_over_detector),
            ("LEAVE_CONFIRM", self.leave_confirm_detector),
            ("PAUSE_MENU", self.pause_menu_detector),
            ("MAIN_MENU", self.main_menu_detector),
        ]

        self._print_health_summary()

    def _print_health_summary(self):

        print("\n========== DETECTOR HEALTH ==========")
        print(f"Use Keys templates     : {len(self.use_keys_detector.templates)}")
        print(f"Game Over templates    : {len(self.game_over_detector.templates)}")
        print(f"Leave confirm templates: {len(self.leave_confirm_detector.templates)}")
        print(f"Pause menu templates   : {len(self.pause_menu_detector.templates)}")
        print(f"Main Menu templates    : {len(self.main_menu_detector.templates)}")
        print("=====================================\n")

    def reset_episode(self):

        self.last_debug = {}
        self.pause_missing_frames = 0

    def detect(self, frame, context="all"):
        import time
        self.last_debug = {
            state.lower(): {
                "score": 0.0,
                "template": None,
                "folder": None,
                "matched": False,
                "votes": 0,
                "detect_ms": 0.0,
            }
            for state, _ in self.detectors
        }
        all_scores = {}

        for state, detector in self.detectors:

            if context == "gameplay":
                # In gameplay context, we only want to run the FAST pause button check
                # to see if the game has ended or crashed.
                if state == "PAUSE_BUTTON":
                    t0 = time.perf_counter()
                    matched, votes, scores = detector.detect(frame)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    
                    all_scores[state] = scores
                    self.last_debug[state.lower()] = {
                        **detector.last_match,
                        "matched": matched,
                        "votes": votes,
                        "detect_ms": elapsed_ms,
                    }
                    
                    if not matched:
                        # 1-frame debounce to instantly trigger death on missing pause button
                        self.pause_missing_frames += 1
                        if self.pause_missing_frames >= 1:
                            return "GAME_OVER_UI", 0, all_scores
                    else:
                        self.pause_missing_frames = 0
                        
                continue

            # Standard context="all" detection
            t0 = time.perf_counter()
            matched, votes, scores = detector.detect(frame)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            all_scores[state] = scores

            self.last_debug[state.lower()] = {
                **detector.last_match,
                "matched": matched,
                "votes": votes,
                "detect_ms": elapsed_ms,
            }

            if matched:
                return state, votes, all_scores

        return "RUNNING", 0, all_scores
