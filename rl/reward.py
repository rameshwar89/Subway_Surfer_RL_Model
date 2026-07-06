from controller.actions import SubwayActions

class RewardSystem:

    SURVIVAL_REWARD = 0.12
    LEFT_PENALTY = -0.030
    RIGHT_PENALTY = -0.030
    JUMP_PENALTY = -0.050
    ROLL_PENALTY = -0.040
    GAME_OVER_PENALTY = -10.0

    def reset(self):
        pass

    def compute(
        self,
        state,
        action,
        episode_steps,
    ):

        if state == "GAME_OVER":
            return self.GAME_OVER_PENALTY

        reward = self.SURVIVAL_REWARD

        # Small survival bonus that grows with time
        reward += min(
            episode_steps * 0.0003,
            0.03,
        )

        # Discourage unnecessary active actions (incentivizes IDLE)
        if action == SubwayActions.LEFT:
            reward += self.LEFT_PENALTY
        elif action == SubwayActions.RIGHT:
            reward += self.RIGHT_PENALTY
        elif action == SubwayActions.JUMP:
            reward += self.JUMP_PENALTY
        elif action == SubwayActions.ROLL:
            reward += self.ROLL_PENALTY

        return reward