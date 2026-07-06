from vision.patch_category_detector import PatchCategoryDetector


class TerminalDetector(PatchCategoryDetector):

    def __init__(self, config):

        super().__init__(
            patch_dir="assets/patches/terminal",
            roi=config["rois"]["terminal"],
            threshold=config["thresholds"]["terminal"],
            min_votes=config["min_votes"]["terminal"],
        )
