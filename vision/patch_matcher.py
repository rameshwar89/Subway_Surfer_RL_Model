import cv2


class PatchMatcher:

    def __init__(self, template_path):

        self.template = cv2.imread(template_path)

        if self.template is None:
            raise FileNotFoundError(template_path)

        self.template_gray = cv2.cvtColor(
            self.template,
            cv2.COLOR_BGR2GRAY,
        )

        self.h, self.w = self.template_gray.shape

    def score(self, frame, x, y):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        roi = gray[
            y:y + self.h,
            x:x + self.w,
        ]

        if roi.shape != self.template_gray.shape:
            return 0.0

        result = cv2.matchTemplate(
            roi,
            self.template_gray,
            cv2.TM_CCOEFF_NORMED,
        )

        _, score, _, _ = cv2.minMaxLoc(result)

        return float(score)