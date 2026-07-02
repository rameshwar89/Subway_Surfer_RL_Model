import cv2


class TemplateMatcher:

    def __init__(self, template_path, threshold=0.95):

        self.threshold = threshold

        self.template = cv2.imread(
            template_path,
            cv2.IMREAD_GRAYSCALE,
        )

        if self.template is None:
            raise FileNotFoundError(template_path)

    def detect(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(
            gray,
            self.template,
            cv2.TM_CCOEFF_NORMED,
        )

        _, score, _, location = cv2.minMaxLoc(result)

        return score >= self.threshold, score