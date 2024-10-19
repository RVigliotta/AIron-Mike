import numpy as np


class BoxingEnv:
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        self.current_step = 0
        self.done = False
        self.result_actions = ['result_win_A', 'result_win_B', 'result_draw']
        self.decision_actions = [col for col in y_data.columns if col not in self.result_actions]

    def reset(self):
        self.current_step = 0
        self.done = False
        return self.X_data[self.current_step]

    def step(self, action):
        reward = 0
        result_action, decision_action = action

        # Predict result first
        if self.y_data.iloc[self.current_step][result_action] == 1:
            reward += 1
            if result_action == 'result_draw':
                # Automatically assign decision_draw for result_draw
                decision_action = 'decision_draw'
                if self.y_data.iloc[self.current_step]['decision_draw'] == 1:
                    reward += 1
                else:
                    reward -= 1
            else:
                # Predict decision next
                if self.y_data.iloc[self.current_step][decision_action] == 1:
                    reward += 1
                else:
                    reward -= 1
        else:
            reward -= 1

        self.current_step += 1
        if self.current_step >= len(self.X_data):
            self.done = True
        next_state = self.X_data[self.current_step] if not self.done else None
        return next_state, reward, self.done

    def sample_action(self):
        possible_result_actions = self.result_actions
        possible_decision_actions = self.decision_actions
        result_action = np.random.choice(possible_result_actions)
        decision_action = 'decision_draw' if result_action == 'result_draw' else np.random.choice(
            possible_decision_actions)
        return result_action, decision_action
