from rl.subway_env import SubwayEnv
import numpy as np

print("Creating environment...")

env = SubwayEnv()

obs, info = env.reset()

print("Environment reset successful!")

print("Observation shape :", obs.shape)
print("Observation dtype :", obs.dtype)
print("Min pixel value   :", obs.min())
print("Max pixel value   :", obs.max())

assert obs.shape == (128, 128, 4)
assert obs.dtype == np.uint8

assert 0 <= obs.min() <= 255
assert 0 <= obs.max() <= 255

for step in range(20):

    print(f"\n========== Step {step+1} ==========")

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    print("Action          :", action)
    print("Reward          :", reward)
    print("State           :", info["state"])
    print("Episode Steps   :", info["episode_steps"])
    print("Observation     :", obs.shape)
    print("Dtype           :", obs.dtype)
    print("Min/Max         :", obs.min(), obs.max())
    print("Terminated      :", terminated)

    assert obs.shape == (128, 128, 4)
    assert obs.dtype == np.uint8

    assert 0 <= obs.min() <= 255
    assert 0 <= obs.max() <= 255

    if terminated:

        print("\nEpisode finished. Resetting...\n")

        obs, info = env.reset()

        assert obs.shape == (128, 128, 4)
        assert obs.dtype == np.uint8

print("\n✅ Gym environment test passed!")

env.close()