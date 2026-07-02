import random
import time

from rl.subway_env import SubwayEnv

env = SubwayEnv()

episode = 1

obs = env.reset()

steps = 0

while True:

    action = random.randint(0, 4)

    obs, reward, done, info = env.step(action)

    steps += 1

    print(
        f"\rEpisode {episode} | Steps {steps} | {info['state']}",
        end=""
    )

    if done:

        print(f"\nEpisode {episode} finished after {steps} steps.")

        episode += 1
        steps = 0

        obs = env.reset()