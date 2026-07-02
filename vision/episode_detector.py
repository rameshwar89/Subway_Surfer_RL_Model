from vision.template_matcher import TemplateMatcher


class EpisodeDetector:

    def __init__(self):

        self.matcher = TemplateMatcher(
            "vision/templates/play.png",
            threshold=0.75,
        )

    def detect(self, frame):

        found, score, location, size = self.matcher.detect(frame)

        return {
            "found": found,
            "score": score,
            "location": location,
            "size": size,
        }

    def is_game_over(self, frame):

        return self.detect(frame)["found"]