from stable_baselines3 import PPO

from rl.subway_env import SubwayEnv


MODEL_PATH = "best_model/best_model.zip"

NUM_EPISODES = 10


def main():

    print("=" * 60)
    print("Subway Surfers PPO Evaluation")
    print("=" * 60)

    env = SubwayEnv()

    print(f"\nLoading model: {MODEL_PATH}")

    model = PPO.load(
        MODEL_PATH,
        env=env,
    )

    episode_rewards = []
    episode_lengths = []

    for episode in range(NUM_EPISODES):

        obs, _ = env.reset()

        done = False

        total_reward = 0.0
        steps = 0

        while not done:

            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        print(
            f"Episode {episode + 1:02d} | "
            f"Reward = {total_reward:.2f} | "
            f"Length = {steps}"
        )

    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    print(f"Episodes              : {NUM_EPISODES}")
    print(f"Average Reward        : {sum(episode_rewards)/NUM_EPISODES:.2f}")
    print(f"Average Length        : {sum(episode_lengths)/NUM_EPISODES:.2f}")

    print(f"Best Reward           : {max(episode_rewards):.2f}")
    print(f"Worst Reward          : {min(episode_rewards):.2f}")

    print(f"Best Episode Length   : {max(episode_lengths)}")
    print(f"Worst Episode Length  : {min(episode_lengths)}")

    env.close()


if __name__ == "__main__":
    main()