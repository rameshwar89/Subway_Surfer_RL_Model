import json
from pathlib import Path
from shutil import copyfile
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.subway_env import SubwayEnv
from train.callbacks import get_callbacks


# --------------------------------------------------
# Run Config
# --------------------------------------------------

# Change this before each run:
# "best"       -> continue from best_model/best_model.zip
# "final"      -> continue from models/subway_ppo_final.zip
# "checkpoint" -> continue from CHECKPOINT_MODEL_PATH
# "scratch"    -> create a new PPO model
TRAIN_FROM = "checkpoint"

FINAL_MODEL_PATH = Path("models/subway_ppo_final.zip")
BEST_MODEL_PATH = Path("best_model/best_model.zip")
BEST_STATS_PATH = Path("best_model/best_stats.json")
CHECKPOINT_MODEL_PATH = Path("models/experiments/Phase18_Scratch_100k_SpeedRun_20260705_210711_complete_130304_20260705_231209.zip")

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_NAME = "Phase21_SpeedRun_Resume_100k_20260706_233252_1"
TOTAL_TIMESTEPS = 35000
SB3_VERBOSE = 1

# --------------------------------------------------
# Create folders
# --------------------------------------------------

Path("models").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)
Path("best_model").mkdir(exist_ok=True)
Path("eval_logs").mkdir(exist_ok=True)


# --------------------------------------------------
# Environment
# --------------------------------------------------

def make_env():
    return Monitor(SubwayEnv())


env = DummyVecEnv([make_env])


# --------------------------------------------------
# Device
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nUsing device: {device}")

if device == "cuda":
    print(torch.cuda.get_device_name(0))


# --------------------------------------------------
# PPO
# --------------------------------------------------

def read_initial_best_score():

    if not BEST_STATS_PATH.exists():
        return None

    with BEST_STATS_PATH.open("r") as f:
        stats = json.load(f)

    return stats.get("best_score")


def get_load_path():

    paths = {
        "best": BEST_MODEL_PATH,
        "final": FINAL_MODEL_PATH,
        "checkpoint": CHECKPOINT_MODEL_PATH,
    }

    if TRAIN_FROM == "scratch":
        return None

    if TRAIN_FROM not in paths:
        raise ValueError(f"Unknown TRAIN_FROM: {TRAIN_FROM}")

    path = paths[TRAIN_FROM]

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot train from {TRAIN_FROM}; missing model: {path}"
        )

    return path


load_path = get_load_path()

if load_path is not None:

    print(f"\nLoading {TRAIN_FROM} model:")
    print(load_path)

    model = PPO.load(
        load_path,
        env=env,
        device=device,
        ent_coef=0.01,
        n_steps=512,
        batch_size=32,
    )
    model.verbose = SB3_VERBOSE

else:

    print("\nCreating New Model...\n")

    model = PPO(
        policy="CnnPolicy",
        env=env,
        device=device,
        verbose=SB3_VERBOSE,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=512,
        batch_size=32,
        n_epochs=10,
        ent_coef=0.01,
        tensorboard_log="logs",
    )


# --------------------------------------------------
# Train
# --------------------------------------------------

interrupted = False

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=get_callbacks(
            initial_best_score=read_initial_best_score(),
        ),
        tb_log_name=EXPERIMENT_NAME,
        reset_num_timesteps=True,
    )

except KeyboardInterrupt:
    interrupted = True
    print("\nTraining interrupted. Saving current model...")


# --------------------------------------------------
# Save Final Model
# --------------------------------------------------

# Update master model
model.save(FINAL_MODEL_PATH.with_suffix(""))

# Save experiment snapshot
snapshot_kind = (
    "interrupted"
    if interrupted
    else "complete"
)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_path = (
    f"models/experiments/"
    f"{EXPERIMENT_NAME}_{snapshot_kind}_"
    f"{model.num_timesteps}_{timestamp}.zip"
)
Path("models/experiments").mkdir(
    parents=True,
    exist_ok=True,
)
copyfile(
    "models/subway_ppo_final.zip",
    experiment_path,
)

print("\nSaved experiment to:")
print(experiment_path)

if interrupted:
    print("Best model was not overwritten unless a new best episode was reached.")
