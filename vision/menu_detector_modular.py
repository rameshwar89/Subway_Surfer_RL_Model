from vision.patch_category_detector import PatchCategoryDetector


class GameOverUIDetector(PatchCategoryDetector):

    def __init__(self, config):

        super().__init__(
            patch_dir="assets/patches/ui/game_over",
            roi=config["rois"]["game_over_ui"],
            threshold=config["thresholds"]["game_over_ui"],
            min_votes=config["min_votes"]["game_over_ui"],
        )


class MainMenuDetector(PatchCategoryDetector):

    def __init__(self, config):

        super().__init__(
            patch_dir="assets/patches/ui/main_menu",
            roi=config["rois"]["main_menu"],
            threshold=config["thresholds"]["main_menu"],
            min_votes=config["min_votes"]["main_menu"],
        )


class LeaveConfirmDetector(PatchCategoryDetector):

    def __init__(self, config, full_config):

        super().__init__(
            patch_dir="assets/patches/ui/leave_confirm",
            roi=full_config.get(
                "leave_confirm_roi",
                config["rois"]["leave_confirm"],
            ),
            threshold=config["thresholds"]["leave_confirm"],
            min_votes=config["min_votes"]["leave_confirm"],
        )
