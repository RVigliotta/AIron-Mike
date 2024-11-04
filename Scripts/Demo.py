import os
import zipfile
import pickle
import pandas as pd
import numpy as np


# Function to replace infinities and NaNs with the column mean
def replace_inf_and_nan(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)
    df = df.infer_objects(copy=False)
    return df


# Function to load zipped models
def load_model_zip(filename):
    directory = '../Models'
    filepath = os.path.join(directory, filename + '.zip')
    with zipfile.ZipFile(filepath, 'r') as zipf:
        with zipf.open(filename) as f:
            return pickle.load(f)


# Loading the models
result_model_rf = load_model_zip('rf_model_result.pkl')
result_model_gb = load_model_zip('gb_model_result.pkl')
decision_model_rf = load_model_zip('rf_model_decision.pkl')
decision_model_gb = load_model_zip('gb_model_decision.pkl')
result_agent = load_model_zip('agent_model_result.pkl')
decision_agent = load_model_zip('agent_model_decision.pkl')

# Prompt for fighter names separately
fighter_A_name = input("Enter the name of fighter A: ")
fighter_B_name = input("Enter the name of fighter B: ")


# Function to request user input
def get_user_input():
    print("Enter fighter data. If a value is not available, type 'null'.")
    data = {}
    prompts = [
        ("the age of fighter A", "age_A"),
        ("the age of fighter B", "age_B"),
        ("the height of fighter A in cm", "height_A"),
        ("the height of fighter B in cm", "height_B"),
        ("the reach of fighter A in cm", "reach_A"),
        ("the reach of fighter B in cm", "reach_B"),
        ("the weight of fighter A in kg", "weight_A"),
        ("the weight of fighter B in kg", "weight_B"),
        ("the number of wins for fighter A", "won_A"),
        ("the number of wins for fighter B", "won_B"),
        ("the number of losses for fighter A", "lost_A"),
        ("the number of losses for fighter B", "lost_B"),
        ("the number of draws for fighter A", "drawn_A"),
        ("the number of draws for fighter B", "drawn_B"),
        ("the number of KOs by fighter A", "kos_A"),
        ("the number of KOs by fighter B", "kos_B"),
        ("the stance of fighter A (orthodox or southpaw)", "stance_A_orthodox", "stance_A_southpaw"),
        ("the stance of fighter B (orthodox o southpaw)", "stance_B_orthodox", "stance_B_southpaw")
    ]
    for prompt in prompts:
        while True:
            value = input(f"Enter {prompt[0]}: ")
            if len(prompt) == 3:
                if value.lower() in ['orthodox', 'southpaw']:
                    data[prompt[1]] = 1 if value.lower() == 'orthodox' else 0
                    data[prompt[2]] = 1 if value.lower() == 'southpaw' else 0
                    break
                else:
                    print("Invalid input. Please enter 'orthodox' or 'southpaw'.")
            else:
                try:
                    data[prompt[1]] = None if value.lower() == 'null' else float(value)
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")

    # Calculate derived features
    data['experience_A'] = data['won_A'] + data['lost_A'] + data['drawn_A']
    data['experience_B'] = data['won_B'] + data['lost_B'] + data['drawn_B']
    data['win_percentage_A'] = data['won_A'] / data['experience_A'] if data['experience_A'] > 0 else 0
    data['win_percentage_B'] = data['won_B'] / data['experience_B'] if data['experience_B'] > 0 else 0
    data['ko_percentage_A'] = data['kos_A'] / data['won_A'] if data['won_A'] > 0 else 0
    data['ko_percentage_B'] = data['kos_B'] / data['won_B'] if data['won_B'] > 0 else 0

    # Add columns for result and decision, initially with null values
    result_actions = ['result_draw', 'result_win_A', 'result_win_B']
    for action in result_actions:
        data[action] = None
    decision_actions = ['decision_DQ', 'decision_KO', 'decision_MD', 'decision_PTS',
                        'decision_RTD', 'decision_SD', 'decision_TD', 'decision_TKO',
                        'decision_UD', 'decision_draw']
    for action in decision_actions:
        data[action] = None
    return data


# Request user input
fighter_data = get_user_input()

# Convert data to DataFrame
fighter_df = pd.DataFrame([fighter_data])

# Data preprocessing
fighter_df = replace_inf_and_nan(fighter_df)

# Ensure DataFrame has the same columns as the model
model_features = result_model_rf.feature_names_in_
fighter_df = fighter_df.reindex(columns=model_features, fill_value=0)


# Function to predict results
def predict_result(model, decision_model, data):
    result_pred = model.predict(data)
    decision_pred = []
    for result in result_pred:
        if result == 'result_draw':
            decision_pred.append('decision_draw')
        else:
            decision_pred.append(decision_model.predict(data)[0])
    return result_pred, decision_pred


# Function to generate a discursive result
def generate_discursive_result(model_name, fighter_A_name, fighter_B_name, result_pred, decision_pred):
    result_map = {
        'result_win_A': f'{fighter_A_name} will win',
        'result_win_B': f'{fighter_B_name} will win',
        'result_draw': f'The match will be a draw'
    }

    decision_map = {
        'decision_SD': 'by split decision',
        'decision_UD': 'by unanimous decision',
        'decision_KO': 'by knockout',
        'decision_TKO': 'by technical knockout',
        'decision_MD': 'by majority decision',
        'decision_RTD': 'by retirement',
        'decision_TD': 'by technical decision',
        'decision_PTS': 'by points',
        'decision_DQ': 'by disqualification',
        'decision_draw': ''
    }

    result_phrase = result_map.get(result_pred[0], 'unknown result')
    decision_phrase = decision_map.get(decision_pred[0], 'unknown decision')

    return f"According to {model_name}, {result_phrase} {decision_phrase}"


# Function to predict using reinforcement agents
def predict_using_agents(result_agent, decision_agent, data):
    result_predictions = []
    decision_predictions = []
    for state in data:
        result_action = result_agent.choose_action(state)
        result_predictions.append(result_action)
        if result_action == 'result_draw':
            decision_action = 'decision_draw'
        else:
            decision_action = decision_agent.choose_action(state, result_action)
        decision_predictions.append(decision_action)
    return result_predictions, decision_predictions


models = {
    'RandomForest': {
        'Result': result_model_rf,
        'Decision': decision_model_rf
    },
    'GradientBoosting': {
        'Result': result_model_gb,
        'Decision': decision_model_gb
    }
}

# Apply models and predict
for model_name, model in models.items():
    print(f"Predictions using {model_name}:")
    result_pred, decision_pred = predict_result(model['Result'], model['Decision'], fighter_df)
    print(f"Result prediction: {result_pred}")
    print(f"Decision prediction: {decision_pred}")
    print(f"Fighter A: {fighter_A_name}, Fighter B: {fighter_B_name}")
    discursive_result = generate_discursive_result(model_name, fighter_A_name, fighter_B_name, result_pred,
                                                   decision_pred)
    print(discursive_result)
    print("\n")

# Predictions using reinforcement agents
result_pred_agent, decision_pred_agent = predict_using_agents(result_agent, decision_agent, fighter_df.values)
print(f"Predictions using ReinforcementAgent:")
print(f"Result prediction: {result_pred_agent}")
print(f"Decision prediction: {decision_pred_agent}")
print(f"Fighter A: {fighter_A_name}, Fighter B: {fighter_B_name}")
discursive_result_agent = generate_discursive_result('ReinforcementAgent', fighter_A_name, fighter_B_name, result_pred_agent, decision_pred_agent)
print(discursive_result_agent)
print("\n")
