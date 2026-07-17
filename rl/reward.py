from controller.actions import SubwayActions

class RewardSystem:

    SURVIVAL_REWARD      = 0.2
    GAME_OVER_PENALTY    = -10.0
    STUMBLE_PENALTY      = 0.0   # on hold
    INVALID_MOVE_PENALTY = -1.0

    # Barrier technique rewards
    CORRECT_ACTION_REWARD = 1.5
    WRONG_ACTION_PENALTY  = -1.0

    # Train dodge reward — only when train is in the sweet spot (0.55–0.65)
    TRAIN_DODGE_REWARD = 1.5

    # Idle bonus when lane is clear
    IDLE_CLEAR_REWARD = 0.1

    def reset(self):
        pass

    def compute(
        self,
        state,
        action,
        episode_steps,
        police_present=False,
        stumbled=False,
        agent_lane=1,
        new_agent_lane=1,
        current_lane_distance=1.0,
        current_lane_type=0.0,
        previous_lane_distance=1.0,
        previous_lane_type=0.0,
        action_active=False,
        on_train=False,
        previous_action=None,
    ):
        breakdown = {
            "stumble_pen": 0.0,
            "invalid_move_pen": 0.0,
            "action_reaction": 0.0,
            "pathfinding": 0.0,
            "animation_tax": 0.0,
            "unnecessary_action": 0.0,
        }

        if state == "GAME_OVER":
            return self.GAME_OVER_PENALTY, breakdown

        reward = self.SURVIVAL_REWARD

        # -------------------------------------------------------------------
        # STUMBLE
        # -------------------------------------------------------------------
        if stumbled:
            reward += self.STUMBLE_PENALTY
            breakdown["stumble_pen"] = self.STUMBLE_PENALTY

        # -------------------------------------------------------------------
        # INVALID MOVE — hit boundary wall
        # -------------------------------------------------------------------
        if action == SubwayActions.LEFT and agent_lane == 0:
            reward += self.INVALID_MOVE_PENALTY
            breakdown["invalid_move_pen"] += self.INVALID_MOVE_PENALTY
        elif action == SubwayActions.RIGHT and agent_lane == 2:
            reward += self.INVALID_MOVE_PENALTY
            breakdown["invalid_move_pen"] += self.INVALID_MOVE_PENALTY

        # -------------------------------------------------------------------
        # IDLE IN CLEAR LANE — reward patience
        # Only when current lane is completely clear (type 0.0, distance >= 0.75)
        # -------------------------------------------------------------------
        if (action == SubwayActions.IDLE
                and previous_lane_type == 0.0
                and previous_lane_distance >= 0.75):
            reward += self.IDLE_CLEAR_REWARD
            breakdown["unnecessary_action"] += self.IDLE_CLEAR_REWARD

        # -------------------------------------------------------------------
        # BARRIER TECHNIQUE — rewards/penalizes correct obstacle technique
        # Fires when a barrier obstacle is in the actionable zone (0.50–0.85).
        # LEFT/RIGHT are never touched here — dodging is always valid.
        # -------------------------------------------------------------------
        if 0.50 <= previous_lane_distance <= 0.85:
            proximity_scale = 1.0 - ((previous_lane_distance - 0.50) / 0.35)

            if previous_lane_type == 0.3:       # BLOCKER — JUMP or ROLL both work
                if action in (SubwayActions.JUMP, SubwayActions.ROLL):
                    bonus = self.CORRECT_ACTION_REWARD * proximity_scale
                    reward += bonus
                    breakdown["action_reaction"] += bonus

            elif previous_lane_type == 0.4:     # ROLL-BLOCKER — JUMP only
                if action == SubwayActions.JUMP:
                    bonus = self.CORRECT_ACTION_REWARD * proximity_scale
                    reward += bonus
                    breakdown["action_reaction"] += bonus
                elif action == SubwayActions.ROLL:
                    reward += self.WRONG_ACTION_PENALTY
                    breakdown["action_reaction"] += self.WRONG_ACTION_PENALTY

            elif previous_lane_type == 0.6:     # JUMP-BARRIER — ROLL only
                if action == SubwayActions.ROLL:
                    bonus = self.CORRECT_ACTION_REWARD * proximity_scale
                    reward += bonus
                    breakdown["action_reaction"] += bonus
                elif action == SubwayActions.JUMP:
                    reward += self.WRONG_ACTION_PENALTY
                    breakdown["action_reaction"] += self.WRONG_ACTION_PENALTY

        # -------------------------------------------------------------------
        # TRAIN DODGE REWARD
        # LEFT or RIGHT rewarded ONLY when:
        #   - There is a solid train (type 1.0) in the current lane
        #   - It is within the sweet spot: 0.55 <= distance <= 0.65
        #   - The agent actually succeeded in changing lane
        # Outside this window, L/R get no bonus and no penalty.
        # -------------------------------------------------------------------
        if (action in (SubwayActions.LEFT, SubwayActions.RIGHT)
                and previous_lane_type == 1.0
                and 0.55 <= previous_lane_distance <= 0.65
                and agent_lane != new_agent_lane):
            reward += self.TRAIN_DODGE_REWARD
            breakdown["action_reaction"] += self.TRAIN_DODGE_REWARD

        # -------------------------------------------------------------------
        # UNNECESSARY ACTION — JUMP/ROLL on a completely clear lane
        # -------------------------------------------------------------------
        if action in (SubwayActions.JUMP, SubwayActions.ROLL) and not on_train:
            if previous_lane_distance >= 0.85 and previous_lane_type == 0.0:
                reward -= 0.3
                breakdown["unnecessary_action"] -= 0.3

        # Small time-alive bonus, capped
        reward += min(episode_steps * 0.0003, 0.05)

        return reward, breakdown