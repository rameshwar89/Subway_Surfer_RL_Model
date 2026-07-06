from datetime import datetime
import json
from pathlib import Path
import shutil

from stable_baselines3.common.callbacks import BaseCallback


class SubwayStatsCallback(BaseCallback):

    STATS_PATH = Path("best_model/best_stats.json")

    def __init__(self, initial_best_score=None):

        super().__init__()

        self.best_episode = 0
        self.best_reward = -1e9
        self.best_score = -1e9

        self._load_previous_best(initial_best_score)

        self.game_overs = 0
        self.total_steps = 0

        # Action distribution
        self.action_counter = {
            0: 0,  # Left
            1: 0,  # Right
            2: 0,  # Jump
            3: 0,  # Roll
            4: 0,  # Idle
        }

        Path("best_model").mkdir(exist_ok=True)

    def _load_previous_best(self, initial_best_score):

        if self.STATS_PATH.exists():
            with self.STATS_PATH.open("r") as f:
                stats = json.load(f)

            self.best_score = stats.get("best_score", self.best_score)
            self.best_reward = stats.get("best_reward", self.best_reward)
            self.best_episode = stats.get("best_episode", self.best_episode)

        if initial_best_score is not None:
            self.best_score = max(self.best_score, initial_best_score)
            self.best_reward = max(self.best_reward, initial_best_score)

    def _save_best_stats(self, score, ep_reward, ep_len, model_path):

        stats = {
            "best_score": score,
            "best_reward": ep_reward,
            "best_episode": ep_len,
            "model_path": model_path,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }

        with self.STATS_PATH.open("w") as f:
            json.dump(stats, f, indent=4)

    def _save_best(self, score, ep_reward, ep_len):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        tmp_path = "best_model/_tmp_best"

        self.model.save(tmp_path)

        destination = (
            f"best_model/"
            f"best_{timestamp}.zip"
        )
        latest_best_path = "best_model/best_model.zip"

        shutil.move(
            tmp_path + ".zip",
            destination,
        )
        shutil.copyfile(
            destination,
            latest_best_path,
        )
        self._save_best_stats(
            score,
            ep_reward,
            ep_len,
            latest_best_path,
        )

        print(
            f"\n*** Best Model Saved ***\n"
            f"Reward : {ep_reward:.2f}\n"
            f"Length : {ep_len}\n"
            f"File   : {destination}\n"
            f"Latest : {latest_best_path}"
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
                self.action_counter[4] / total_actions,
            )

            self.logger.record(
                "policy/left",
                self.action_counter[0] / total_actions,
            )

            self.logger.record(
                "policy/right",
                self.action_counter[1] / total_actions,
            )

            self.logger.record(
                "policy/jump",
                self.action_counter[2] / total_actions,
            )

            self.logger.record(
                "policy/roll",
                self.action_counter[3] / total_actions,
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
