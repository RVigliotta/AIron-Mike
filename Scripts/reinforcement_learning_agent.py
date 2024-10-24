import os
import pickle
import time
import zipfile

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from Scripts.environment import ResultEnv, ResultAgent, DecisionEnv, DecisionAgent


def replace_inf_and_nan(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df


def load_and_preprocess_data():
    train_X = pd.read_csv('../data/splitted/X_train.csv')
    train_y = pd.read_csv('../data/splitted/y_train.csv')
    test_X = pd.read_csv('../data/splitted/X_test.csv')
    test_y = pd.read_csv('../data/splitted/y_test.csv')
    val_X = pd.read_csv('../data/splitted/X_val.csv')
    val_y = pd.read_csv('../data/splitted/y_val.csv')
    for df in [train_X, val_X, test_X]:
        replace_inf_and_nan(df)
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)
    result_actions = ['result_win_A', 'result_win_B', 'result_draw']
    decision_actions = [col for col in train_y.columns if col not in result_actions]
    train_y_result = train_y[result_actions]
    test_y_result = test_y[result_actions]
    val_y_result = val_y[result_actions]
    train_y_decision = train_y[decision_actions]
    test_y_decision = test_y[decision_actions]
    val_y_decision = val_y[decision_actions]
    return train_X, train_y_result, test_X, test_y_result, val_X, val_y_result, train_y_decision, test_y_decision, val_y_decision


def train_agent(env, agent, num_episodes, result_agent=None):
    start_time = time.time()
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        while not env.done:
            if isinstance(agent, DecisionAgent) and result_agent:
                result_action = result_agent.choose_action(state)
                action = agent.choose_action(state, result_action)
            else:
                action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            agent.update_epsilon()
            state = next_state
            total_reward += reward
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
    elapsed_time = time.time() - start_time
    print(f"Training time: {elapsed_time:.2f} seconds")


def save_model_zip(agent, filename):
    directory = '../models'
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with zipfile.ZipFile(filepath + '.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        with zipf.open(filename, 'w') as f:
            pickle.dump(agent, f)


def load_model_zip(filename):
    directory = '../models'
    filepath = os.path.join(directory, filename + '.zip')
    with zipfile.ZipFile(filepath, 'r') as zipf:
        with zipf.open(filename) as f:
            return pickle.load(f)


def evaluate_model(agent, test_X, test_y, actions, result_agent=None):
    predicted_labels = []
    for state in test_X:
        if isinstance(agent, DecisionAgent) and result_agent:
            result_action = result_agent.choose_action(state)
            action = agent.choose_action(state, result_action)
        else:
            action = agent.choose_action(state)
        pred = [1 if act == action else 0 for act in actions]
        predicted_labels.append(pred)
    y_true = test_y.values
    predicted_labels = np.array(predicted_labels)
    if predicted_labels.shape == y_true.shape:
        accuracy = accuracy_score(y_true, predicted_labels)
        precision = precision_score(y_true, predicted_labels, average='weighted', zero_division=0)
        recall = recall_score(y_true, predicted_labels, average='weighted', zero_division=0)
        f1 = f1_score(y_true, predicted_labels, average='weighted', zero_division=0)
        try:
            roc_auc = roc_auc_score(y_true, predicted_labels, average='weighted', multi_class='ovr')
        except ValueError:
            roc_auc = None
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("ROC-AUC:", roc_auc)
    else:
        print("Shape mismatch between y_true and predicted_labels.")


def main():
    train_X, train_y_result, test_X, test_y_result, val_X, val_y_result, train_y_decision, test_y_decision, val_y_decision = load_and_preprocess_data()

    result_actions = ['result_win_A', 'result_win_B', 'result_draw']
    decision_actions = [col for col in train_y_decision.columns]

    # Training the Result Agent
    result_env = ResultEnv(train_X, train_y_result, result_actions)
    result_agent = ResultAgent(result_env)
    train_agent(result_env, result_agent, 10)  # Increase the number of episodes for more complete training
    save_model_zip(result_agent, 'result_agent_model.pkl')

    # Evaluation of the Result Agent
    evaluate_model(result_agent, test_X, test_y_result, result_actions)

    # Generate predictions from the Result Agent for the training set
    predicted_train_results = []
    for state in train_X:
        result_action = result_agent.choose_action(state)
        predicted_train_results.append(result_action)

    # Use the predictions of the Result Agent as input for the Decision Agent
    train_X_decision = np.concatenate((train_X, np.array(predicted_train_results).reshape(-1, 1)), axis=1)
    decision_env = DecisionEnv(train_X_decision, train_y_decision, decision_actions)
    decision_agent = DecisionAgent(decision_env)

    # Training the Decision Agent
    train_agent(decision_env, decision_agent, 20, result_agent)
    save_model_zip(decision_agent, 'decision_agent_model.pkl')

    # Evaluation of the Decision Agent
    predicted_test_results = []
    for state in test_X:
        result_action = result_agent.choose_action(state)
        predicted_test_results.append(result_action)
    test_X_decision = np.concatenate((test_X, np.array(predicted_test_results).reshape(-1, 1)), axis=1)
    decision_env.X_data = test_X_decision

    evaluate_model(decision_agent, test_X_decision, test_y_decision, decision_actions, result_agent)


if __name__ == "__main__":
    main()
