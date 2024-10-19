import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from Scripts.environment import BoxingEnv


# Replace infinities and NaN with the column mean
def replace_inf_and_nan(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df


# Load split data
train_X = pd.read_csv('../data/splitted/X_train.csv')
train_y = pd.read_csv('../data/splitted/y_train.csv')
test_X = pd.read_csv('../data/splitted/X_test.csv')
test_y = pd.read_csv('../data/splitted/y_test.csv')
val_X = pd.read_csv('../data/splitted/X_val.csv')
val_y = pd.read_csv('../data/splitted/y_val.csv')

# Verify the shape of datasets
print(f"Train X shape: {train_X.shape}")
print(f"Train y shape: {train_y.shape}")
print(f"Test X shape: {test_X.shape}")
print(f"Test y shape: {test_y.shape}")
print(f"Validation X shape: {val_X.shape}")
print(f"Validation y shape: {val_y.shape}")

# Replace NaNs and infinities
train_X = replace_inf_and_nan(train_X)
val_X = replace_inf_and_nan(val_X)
test_X = replace_inf_and_nan(test_X)

# Scaling
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
val_X = scaler.transform(val_X)
test_X = scaler.transform(test_X)

# Possible actions (example): win, draw, lose
actions = train_y.columns  # Assuming train_y has columns for result classes
print(f"Available actions: {actions}")

# Initialize the environment
env = BoxingEnv(train_X, train_y)

# Run a simulation for 5 steps as a test
state = env.reset()
for _ in range(5):
    action = env.sample_action()  # Random action
    next_state, reward, done = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Next state: {next_state}")
    if done:
        print("Episode finished!")
        break
