from vision.patch_category_detector import PatchCategoryDetector


class PopupDetector(PatchCategoryDetector):

    def __init__(self, config):

        super().__init__(
            patch_dir="assets/patches/popup/use_keys",
            roi=config["rois"]["popup"],
            threshold=config["thresholds"]["popup"],
            min_votes=config["min_votes"]["popup"],
        )
