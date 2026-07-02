class RewardSystem:

    def compute(self, state):

        if state == "RUNNING":
            return 0.1

        if state == "GAME_OVER":
            return -10

        return 0