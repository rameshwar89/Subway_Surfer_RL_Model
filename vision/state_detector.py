import json

from vision.patch_matcher import PatchMatcher


class StateDetector:

    def __init__(self):

        self.states = {
            "MAIN_MENU": [
                "start_button",
                "settings_icon",
                "shop_icon",
                "logo",
            ],

            "GAME_OVER": [
                "play_button",
                "score_panel",
                "bottom_panel",
            ],

            "USE_KEYS": [
                "use_keys",
            ],
        }

        self.detectors = {}

        for state, patches in self.states.items():

            self.detectors[state] = []

            for patch in patches:

                matcher = PatchMatcher(
                    f"vision/references/{patch}.png"
                )

                with open(
                    f"vision/references/{patch}.json",
                    "r",
                ) as f:

                    meta = json.load(f)

                self.detectors[state].append(
                    {
                        "name": patch,
                        "matcher": matcher,
                        "x": meta["x"],
                        "y": meta["y"],
                    }
                )

    def _votes(self, frame, state):

        votes = 0

        patch_scores = {}

        for patch in self.detectors[state]:

            score = patch["matcher"].score(
                frame,
                patch["x"],
                patch["y"],
            )

            patch_scores[patch["name"]] = score

            if score > 0.90:
                votes += 1

        return votes, patch_scores

    def detect(self, frame):

        menu_votes, menu_scores = self._votes(
            frame,
            "MAIN_MENU",
        )

        game_votes, game_scores = self._votes(
            frame,
            "GAME_OVER",
        )

        use_votes, use_scores = self._votes(
            frame,
            "USE_KEYS",
        )

        if game_votes >= 2:
            return (
                "GAME_OVER",
                game_votes,
                game_scores,
            )

        if use_votes >= 1:
            return (
                "USE_KEYS",
                use_votes,
                use_scores,
            )

        if menu_votes >= 3:
            return (
                "MAIN_MENU",
                menu_votes,
                menu_scores,
            )

        return (
            "RUNNING",
            0,
            {},
        )