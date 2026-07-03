from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
)

from train.stats_callback import SubwayStatsCallback


def get_callbacks():

    checkpoint_callback = CheckpointCallback(
        save_freq=200,
        save_path="models/",
        name_prefix="latest_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    stats_callback = SubwayStatsCallback()

    return CallbackList([
        checkpoint_callback,
        stats_callback,
    ])