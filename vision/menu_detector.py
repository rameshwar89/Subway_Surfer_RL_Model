from vision.template_matcher import TemplateMatcher


class MenuDetector:

    def __init__(self):

        self.matcher = TemplateMatcher(
            "vision/templates/gameover_full.png",
            threshold=0.95,
        )

    def is_game_over(self, frame):

        found, score = self.matcher.detect(frame)

        return found, score