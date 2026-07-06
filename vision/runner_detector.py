from vision.patch_category_detector import PatchCategoryDetector


class RunnerDetector(PatchCategoryDetector):

    def __init__(self, config):

        super().__init__(
            patch_dir="assets/patches/running/runner",
            roi=config["rois"]["runner"],
            threshold=config["thresholds"]["runner"],
            min_votes=config["min_votes"]["runner"],
        )
