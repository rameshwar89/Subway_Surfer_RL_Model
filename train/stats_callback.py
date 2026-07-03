from pathlib import Path
import shutil

from stable_baselines3.common.callbacks import BaseCallback


class SubwayStatsCallback(BaseCallback):

    def __init__(self):

        super().__init__()

        self.best_episode = 0
        self.best_reward = -1e9
        self.best_score = -1e9

        self.game_overs = 0

        Path("best_model").mkdir(exist_ok=True)

    def _save_best(self, score, ep_reward, ep_len):

        tmp_path = "best_model/_tmp_best"

        self.model.save(tmp_path)

        shutil.move(
            tmp_path + ".zip",
            "best_model/best_model.zip",
        )

        print(
            f"\n⭐ Best Model Updated | "
            f"Score: {score:.2f} | "
            f"Reward: {ep_reward:.2f} | "
            f"Length: {ep_len}"
        )

    def _on_step(self):

        infos = self.locals["infos"]

        if len(infos) == 0:
            return True

        info = infos[0]

        # ----------------------------------
        # Episode finished
        # ----------------------------------

        if "episode" in info:

            ep_len = info["episode"]["l"]
            ep_reward = info["episode"]["r"]

            # Combined score
            score = ep_reward

            if score > self.best_score:

                self.best_score = score
                self.best_episode = ep_len
                self.best_reward = ep_reward

                self._save_best(
                    score,
                    ep_reward,
                    ep_len,
                )

            self.logger.record(
                "custom/episode_length",
                ep_len,
            )

            self.logger.record(
                "custom/episode_reward",
                ep_reward,
            )

            self.logger.record(
                "custom/best_episode",
                self.best_episode,
            )

            self.logger.record(
                "custom/best_reward",
                self.best_reward,
            )

            self.logger.record(
                "custom/best_score",
                self.best_score,
            )

        # ----------------------------------
        # Performance
        # ----------------------------------

        if "step_time" in info:

            self.logger.record(
                "performance/step_time_ms",
                info["step_time"] * 1000,
            )

            self.logger.record(
                "performance/steps_per_second",
                1 / info["step_time"],
            )

        # ----------------------------------
        # Game Overs
        # ----------------------------------

        state = info.get("state")

        if state == "GAME_OVER":
            self.game_overs += 1

        self.logger.record(
            "custom/game_overs",
            self.game_overs,
        )

        return True