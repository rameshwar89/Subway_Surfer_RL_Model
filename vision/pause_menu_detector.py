from vision.patch_category_detector import PatchCategoryDetector


class PauseMenuDetector(PatchCategoryDetector):

    def __init__(self, config, full_config):

        super().__init__(
            patch_dir="assets/patches/ui/pause_menu",
            roi=full_config.get(
                "pause_menu_roi",
                config["rois"]["pause_menu"],
            ),
            threshold=config["thresholds"]["pause_menu"],
            min_votes=config["min_votes"]["pause_menu"],
        )
