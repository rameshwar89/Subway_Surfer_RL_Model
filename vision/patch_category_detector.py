import json
from pathlib import Path

import cv2


class PatchCategoryDetector:

    def __init__(
        self,
        patch_dir,
        roi,
        threshold,
        min_votes=1,
    ):

        self.patch_dir = Path(patch_dir)
        self.roi = roi
        self.threshold = threshold
        self.min_votes = min_votes
        self.last_match = {
            "score": 0.0,
            "template": None,
            "folder": None,
        }
        self.templates = self._load_templates()

    def _load_templates(self):

        templates = []

        if not self.patch_dir.exists():
            return templates

        for path in sorted(self.patch_dir.rglob("*.png")):

            template = cv2.imread(
                str(path),
                cv2.IMREAD_GRAYSCALE,
            )

            if template is None:
                continue

            meta_path = path.with_suffix(".json")
            meta = None

            if meta_path.exists():
                with meta_path.open("r") as f:
                    meta = json.load(f)

            templates.append(
                {
                    "name": path.stem,
                    "template": template,
                    "path": str(path),
                    "folder": path.parent.name,
                    "meta": meta,
                }
            )

        return templates

    def _crop(self, frame):

        h, w = frame.shape[:2]

        x = max(0, self.roi["x"])
        y = max(0, self.roi["y"])
        x2 = min(w, x + self.roi["width"])
        y2 = min(h, y + self.roi["height"])

        return frame[y:y2, x:x2]

    def _fixed_score(self, gray, template, meta):

        x = meta["x"] - self.roi["x"]
        y = meta["y"] - self.roi["y"]
        h, w = template.shape

        if (
            x < 0
            or y < 0
            or x + w > gray.shape[1]
            or y + h > gray.shape[0]
        ):
            return 0.0

        patch = gray[
            y:y+h,
            x:x+w,
        ]

        if patch.shape != template.shape:
            return 0.0

        result = cv2.matchTemplate(
            patch,
            template,
            cv2.TM_CCOEFF_NORMED,
        )

        _, score, _, _ = cv2.minMaxLoc(result)
        return float(score)
    
    def _local_sliding_score(
        self,
        gray,
        template,
        meta,
        radius=40,
    ):

        cx = meta["x"] - self.roi["x"]
        cy = meta["y"] - self.roi["y"]

        h, w = template.shape

        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)

        x2 = min(gray.shape[1], cx + w + radius)
        y2 = min(gray.shape[0], cy + h + radius)

        window = gray[y1:y2, x1:x2]

        if (
            window.shape[0] < h
            or window.shape[1] < w
        ):
            return 0.0

        result = cv2.matchTemplate(
            window,
            template,
            cv2.TM_CCOEFF_NORMED,
        )

        _, score, _, _ = cv2.minMaxLoc(result)

        return float(score)

    def _sliding_score(self, gray, template):

        if (
            gray.shape[0] < template.shape[0]
            or gray.shape[1] < template.shape[1]
        ):
            return 0.0

        result = cv2.matchTemplate(
            gray,
            template,
            cv2.TM_CCOEFF_NORMED,
        )

        _, score, _, _ = cv2.minMaxLoc(result)
        return float(score)

    def detect(self, frame):

        if not self.templates:
            self.last_match = {
                "score": 0.0,
                "template": None,
                "folder": None,
            }
            return False, 0, {}

        roi = self._crop(frame)

        if roi.size == 0:
            self.last_match = {
                "score": 0.0,
                "template": None,
                "folder": None,
            }
            return False, 0, {}

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY,
        )

        votes = 0
        scores = {}
        best_score = 0.0
        best_template = None
        best_folder = None

        for item in self.templates:

            template = item["template"]

            if item["meta"] is not None:

                score = self._fixed_score(
                    gray,
                    template,
                    item["meta"],
                )

                # -------------------------------------------------
                # Near miss?
                # Search a small area around the expected position.
                # -------------------------------------------------

                if (
                    score < self.threshold
                    and score >= self.threshold - 0.15
                ):

                    local_score = self._local_sliding_score(
                        gray,
                        template,
                        item["meta"],
                        radius=40,
                    )

                    score = max(
                        score,
                        local_score,
                    )

            else:

                score = self._sliding_score(
                    gray,
                    template,
                )
            scores[item["name"]] = score

            if best_template is None or score > best_score:
                best_score = score
                best_template = item["name"]
                best_folder = item["folder"]

            if score >= self.threshold:
                votes += 1

        self.last_match = {
            "score": best_score,
            "template": best_template,
            "folder": best_folder,
        }

        return votes >= self.min_votes, votes, scores


def load_detection_config(path="configs/calibration.json"):

    with open(path, "r") as f:
        cfg = json.load(f)

    return cfg["state_detection"]


def load_calibration(path="configs/calibration.json"):

    with open(path, "r") as f:
        return json.load(f)
