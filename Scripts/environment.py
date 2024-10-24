import numpy as np
import random


class ResultEnv:
    def __init__(self, X_data, y_data, result_actions):
        self.X_data = X_data
        self.y_data = y_data
        self.current_step = 0
        self.done = False
        self.result_actions = result_actions

    def reset(self):
        self.current_step = 0
        self.done = False
        return self.X_data[self.current_step]

    def step(self, result_action):
        reward = 0

        if self.y_data.iloc[self.current_step][result_action] == 1:
            reward += 1
        else:
            reward -= 1

        self.current_step += 1
        if self.current_step >= len(self.X_data):
            self.done = True

        next_state = self.X_data[self.current_step] if not self.done else None
        return next_state, reward, self.done


class DecisionEnv:
    def __init__(self, X_data, y_data, decision_actions):
        self.X_data = X_data
        self.y_data = y_data
        self.current_step = 0
        self.done = False
        self.decision_actions = decision_actions

    def reset(self):
        self.current_step = 0
        self.done = False
        return self.X_data[self.current_step]

    def step(self, decision_action):
        reward = 0
        if self.y_data.iloc[self.current_step][decision_action] == 1:
            reward += 1
        else:
            reward -= 1
        self.current_step += 1
        if self.current_step >= len(self.X_data):
            self.done = True
        next_state = self.X_data[self.current_step] if not self.done else None
        return next_state, reward, self.done


class ResultAgent:
    def __init__(self, env, alpha=0.01, gamma=0.95, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((len(env.X_data), len(env.result_actions)))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            result_action = random.choice(self.env.result_actions)
        else:
            state_index = self.env.current_step % len(self.q_table)
            action_index = np.argmax(self.q_table[state_index])
            result_action = self.env.result_actions[action_index]
        return result_action

    def learn(self, state, action, reward, next_state):
        state_index = self.env.current_step - 1
        next_state_index = self.env.current_step if not self.env.done else state_index
        result_action_index = self.env.result_actions.index(action)
        best_future_q = np.max(self.q_table[next_state_index])
        current_q = self.q_table[state_index, result_action_index]
        self.q_table[state_index, result_action_index] = (
            current_q + self.alpha * (reward + self.gamma * best_future_q - current_q)
        )

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class DecisionAgent:
    def __init__(self, env, alpha=0.01, gamma=0.95, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((len(env.X_data), len(env.decision_actions)))

    def choose_action(self, state, result_action):
        if result_action == 'result_draw':
            decision_action = 'decision_draw'
        else:
            if np.random.rand() < self.epsilon:
                decision_action = random.choice(self.env.decision_actions)
            else:
                state_index = self.env.current_step % len(self.q_table)
                action_index = np.argmax(self.q_table[state_index])
                decision_action = self.env.decision_actions[action_index]
        return decision_action

    def learn(self, state, action, reward, next_state):
        state_index = self.env.current_step - 1
        next_state_index = self.env.current_step if not self.env.done else state_index
        decision_action_index = self.env.decision_actions.index(action)
        best_future_q = np.max(self.q_table[next_state_index])
        current_q = self.q_table[state_index, decision_action_index]
        self.q_table[state_index, decision_action_index] = (
                current_q + self.alpha * (reward + self.gamma * best_future_q - current_q)
        )

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
