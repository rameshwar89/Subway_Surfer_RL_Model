from rl.subway_env import SubwayEnv

print("Creating environment...")

env = SubwayEnv()

obs, info = env.reset()

print("Environment reset successful!")
print("Observation shape:", obs.shape)

assert obs.shape == (128, 128, 4)

for step in range(20):

    print(f"\nStep {step + 1}")

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    print("Action:", action)
    print("Reward:", reward)
    print("State:", info["state"])
    print("Observation shape:", obs.shape)
    print("Terminated:", terminated)

    assert obs.shape == (128, 128, 4)

    if terminated:

        print("\nEpisode finished. Resetting...\n")

        obs, info = env.reset()

        assert obs.shape == (128, 128, 4)

print("\nGym environment test passed!")

env.close()