import os
import pandas as pd
import numpy as np
import zipfile
import pickle


# Function to replace infinities and NaNs with the mean of the columns
def replace_inf_and_nan(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)
    df = df.infer_objects(copy=False)
    return df


# Function to load models from zip files
def load_model_zip(filename):
    directory = '../Models'
    filepath = os.path.join(directory, filename + '.zip')
    with zipfile.ZipFile(filepath, 'r') as zipf:
        with zipf.open(filename) as f:
            return pickle.load(f)


# Load models
result_model_rf = load_model_zip('rf_model_result.pkl')
result_model_gb = load_model_zip('gb_model_result.pkl')
decision_model_rf = load_model_zip('rf_model_decision.pkl')
decision_model_gb = load_model_zip('gb_model_decision.pkl')

# Ask for fighter names separately
fighter_A_name = input("Inserisci il nome del fighter A: ")
fighter_B_name = input("Inserisci il nome del fighter B: ")


# Function to get user input
def get_user_input():
    print("Inserisci i dati dei fighter. Se un dato non è disponibile, digita 'null'.")
    data = {}
    prompts = [
        ("l'età del fighter A", "age_A"),
        ("l'età del fighter B", "age_B"),
        ("l'altezza del fighter A in cm", "height_A"),
        ("l'altezza del fighter B in cm", "height_B"),
        ("l'apertura alare del fighter A in cm", "reach_A"),
        ("l'apertura alare del fighter B in cm", "reach_B"),
        ("il peso del fighter A in kg", "weight_A"),
        ("il peso del fighter B in kg", "weight_B"),
        ("il numero di vittorie del fighter A", "won_A"),
        ("il numero di vittorie del fighter B", "won_B"),
        ("il numero di sconfitte del fighter A", "lost_A"),
        ("il numero di sconfitte del fighter B", "lost_B"),
        ("il numero di pareggi del fighter A", "drawn_A"),
        ("il numero di pareggi del fighter B", "drawn_B"),
        ("il numero di KO ottenuti dal fighter A", "kos_A"),
        ("il numero di KO ottenuti dal fighter B", "kos_B"),
        ("la stance del fighter A (orthodox o southpaw)", "stance_A_orthodox", "stance_A_southpaw"),
        ("la stance del fighter B (orthodox o southpaw)", "stance_B_orthodox", "stance_B_southpaw")
    ]
    for prompt in prompts:
        value = input(f"Inserisci {prompt[0]}: ")
        if len(prompt) == 3:
            data[prompt[1]] = 1 if value.lower() == 'orthodox' else 0
            data[prompt[2]] = 1 if value.lower() == 'southpaw' else 0
        else:
            data[prompt[1]] = None if value.lower() == 'null' else float(value)

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


# Get user input
fighter_data = get_user_input()

# Convert data to DataFrame
fighter_df = pd.DataFrame([fighter_data])

# Preprocess data
fighter_df = replace_inf_and_nan(fighter_df)

# Ensure DataFrame has the same columns as the model
model_features = result_model_rf.feature_names_in_
fighter_df = fighter_df.reindex(columns=model_features, fill_value=0)


# Function to predict results
def predict_result(model, data):
    return model.predict(data)


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
    result_pred = predict_result(model['Result'], fighter_df)
    decision_pred = predict_result(model['Decision'], fighter_df)
    print(f"Result prediction: {result_pred}")
    print(f"Decision prediction: {decision_pred}")
    print(f"Fighter A: {fighter_A_name}, Fighter B: {fighter_B_name}")
    print("\n")
