import json


class RewardSystem:
    def __init__(self):

        with open("configs/reward.json", "r") as f:
            cfg = json.load(f)

        self.survival_reward = cfg["survival_reward"]
        self.death_penalty = cfg["death_penalty"]

        self.milestone_interval = cfg["milestone_interval"]
        self.milestone_bonus = cfg["milestone_bonus"]

        self.spam_penalty = cfg["spam_penalty"]

        self.reset()

    def reset(self):

        self.last_action = None
        self.repeat_count = 0

    def compute(
        self,
        state,
        action=None,
        episode_steps=0,
    ):

        reward = 0.0

        # -----------------------
        # Survival
        # -----------------------

        if state == "RUNNING":
            reward += self.survival_reward

        # -----------------------
        # Survival milestones
        # -----------------------

        if (
            state == "RUNNING"
            and episode_steps > 0
            and episode_steps % self.milestone_interval == 0
        ):
            reward += self.milestone_bonus

        # -----------------------
        # Game Over
        # -----------------------

        if state == "GAME_OVER":
            reward += self.death_penalty

        # -----------------------
        # Optional action repetition penalty
        # -----------------------

        if self.spam_penalty > 0 and action is not None:

            if action == self.last_action:
                self.repeat_count += 1
            else:
                self.repeat_count = 0

            if self.repeat_count >= 8:
                reward -= self.spam_penalty

            self.last_action = action

        return reward