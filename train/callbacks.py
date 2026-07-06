from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
)

from train.stats_callback import SubwayStatsCallback


def get_callbacks(initial_best_score=None):

    checkpoint_callback = CheckpointCallback(
        save_freq=2000,
        save_path="models/checkpoints/",
        name_prefix="latest_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    stats_callback = SubwayStatsCallback(
        initial_best_score=initial_best_score,
    )

    return CallbackList(
        [
            checkpoint_callback,
            stats_callback,
        ]
    )