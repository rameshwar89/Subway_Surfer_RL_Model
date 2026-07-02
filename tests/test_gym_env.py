from rl.subway_env import SubwayEnv

env = SubwayEnv()

print("Creating environment...")

obs, info = env.reset()

print("Environment reset successful!")
print("Observation shape:", obs.shape)

for i in range(20):

    action = env.action_space.sample()

    print(f"\nStep {i+1}")
    print("Action:", action)

    obs, reward, terminated, truncated, info = env.step(action)

    print("Reward:", reward)
    print("State:", info["state"])
    print("Terminated:", terminated)

    if terminated:
        print("\nEpisode finished. Resetting...\n")
        obs, info = env.reset()

env.close()

print("\nGym environment test passed!")
