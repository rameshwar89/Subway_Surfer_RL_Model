from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
)


def get_callbacks():

    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="models/",
        name_prefix="ppo_subway",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    return CallbackList([
        checkpoint_callback,
    ])