from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.subway_env import SubwayEnv
from train.callbacks import get_callbacks
from stable_baselines3.common.monitor import Monitor


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

best_model = Path("best_model/best_model.zip")

if best_model.exists():

    print("\nLoading Best Model...")

    model = PPO.load(
        best_model,
        env=env,
        device=device,
    )

else:

    print("\nCreating New Model...")

    model = PPO(
        policy="CnnPolicy",
        env=env,
        device=device,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=64,
        batch_size=64,
        n_epochs=10,
        tensorboard_log="logs",
    )


# --------------------------------------------------
# Train
# --------------------------------------------------

model.learn(
    total_timesteps=5000,
    callback=get_callbacks(),
    tb_log_name="SubwayPPO_17",
)


# --------------------------------------------------
# Save Final Model
# --------------------------------------------------

model.save(
    "models/subway_ppo_final"
)

print("\nTraining completed!")