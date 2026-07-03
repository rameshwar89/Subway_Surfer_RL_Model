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
        self.total_steps = 0

        # Action distribution
        self.action_counter = {
            0: 0,  # Idle
            1: 0,  # Left
            2: 0,  # Right
            3: 0,  # Jump
            4: 0,  # Roll
        }

        Path("best_model").mkdir(exist_ok=True)

    def _save_best(self, selection_metric, ep_reward, ep_len):

        tmp_path = "best_model/_tmp_best"

        self.model.save(tmp_path)

        shutil.move(
            tmp_path + ".zip",
            "best_model/best_model.zip",
        )

        print(
            f"\n⭐ Best Model Updated | "
            f"Metric: {selection_metric:.2f} | "
            f"Reward: {ep_reward:.2f} | "
            f"Length: {ep_len}"
        )

    def _on_step(self):

        self.total_steps += 1

        infos = self.locals["infos"]

        if len(infos) == 0:
            return True

        info = infos[0]

        # -------------------------------------------------
        # Episode statistics
        # -------------------------------------------------

        if "episode" in info:

            ep_len = info["episode"]["l"]
            ep_reward = info["episode"]["r"]

            # For now reward is our selection metric.
            # Later this can become real Subway Surfers score.
            selection_metric = ep_reward

            if selection_metric > self.best_score:

                self.best_score = selection_metric
                self.best_episode = ep_len
                self.best_reward = ep_reward

                self._save_best(
                    selection_metric,
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

        # -------------------------------------------------
        # Performance
        # -------------------------------------------------

        if "step_time" in info:

            step_time = info["step_time"]

            self.logger.record(
                "performance/step_time_ms",
                step_time * 1000,
            )

            if step_time > 0:

                self.logger.record(
                    "performance/steps_per_second",
                    1 / step_time,
                )

        # -------------------------------------------------
        # Action distribution
        # -------------------------------------------------

        action = info.get("action")

        if action is not None:

            self.action_counter[action] += 1

        total_actions = sum(self.action_counter.values())

        if total_actions > 0:

            self.logger.record(
                "policy/idle",
                self.action_counter[0] / total_actions,
            )

            self.logger.record(
                "policy/left",
                self.action_counter[1] / total_actions,
            )

            self.logger.record(
                "policy/right",
                self.action_counter[2] / total_actions,
            )

            self.logger.record(
                "policy/jump",
                self.action_counter[3] / total_actions,
            )

            self.logger.record(
                "policy/roll",
                self.action_counter[4] / total_actions,
            )

        # -------------------------------------------------
        # Reward breakdown (future-ready)
        # -------------------------------------------------

        reward_breakdown = info.get("reward_breakdown")

        if reward_breakdown:

            for key, value in reward_breakdown.items():

                self.logger.record(
                    f"reward/{key}",
                    value,
                )

        # -------------------------------------------------
        # Environment state
        # -------------------------------------------------

        state = info.get("state")

        if state == "GAME_OVER":

            self.game_overs += 1

        self.logger.record(
            "custom/game_overs",
            self.game_overs,
        )

        return True