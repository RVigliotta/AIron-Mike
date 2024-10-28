import os
import pandas as pd
import numpy as np
import zipfile
import pickle
from joblib import delayed, Parallel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


# Replace infinities and NaN with the column mean
def replace_inf_and_nan(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df


# Function to load a CSV file
def load_csv(file_path):
    return pd.read_csv(file_path)


def save_model_zip(agent, filename):
    directory = '../models'
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with zipfile.ZipFile(filepath + '.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        with zipf.open(filename, 'w') as f:
            pickle.dump(agent, f)


# Parallel loading of preprocessed and split datasets
file_paths = [
    '../data/splitted/x_train.csv',
    '../data/splitted/y_train.csv',
    '../data/splitted/x_test.csv',
    '../data/splitted/y_test.csv',
    '../data/splitted/x_val.csv',
    '../data/splitted/y_val.csv'
]
train_x, train_y, test_x, test_y, val_x, val_y = Parallel(n_jobs=2)(delayed(load_csv)(file) for file in file_paths)

print(f"Initial train_x dimensions: {train_x.shape}")
print(f"Initial train_y dimensions: {train_y.shape}")

train_x = replace_inf_and_nan(train_x)
val_x = replace_inf_and_nan(val_x)
test_x = replace_inf_and_nan(test_x)

# Outlier analysis
iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(train_x)
mask = yhat != -1
train_x_no_outliers = train_x[mask]
train_y_no_outliers = train_y[mask]

print(f"train_x dimensions without outliers: {train_x_no_outliers.shape}")
print(f"train_y dimensions without outliers: {train_y_no_outliers.shape}")

if train_y_no_outliers.ndim > 1:
    train_y_no_outliers = train_y_no_outliers.iloc[:, 0].values.ravel()

# Apply SMOTE
smote = SMOTE(random_state=42)
train_x_resampled, train_y_resampled = smote.fit_resample(train_x_no_outliers, train_y_no_outliers)

# Preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95))
])
train_x_preprocessed = preprocessing_pipeline.fit_transform(train_x_resampled)
val_x_preprocessed = preprocessing_pipeline.transform(val_x)
test_x_preprocessed = preprocessing_pipeline.transform(test_x)

train_x_preprocessed = pd.DataFrame(train_x_preprocessed,
                                    columns=[f'PC{i + 1}' for i in range(train_x_preprocessed.shape[1])])
val_x_preprocessed = pd.DataFrame(val_x_preprocessed,
                                  columns=[f'PC{i + 1}' for i in range(val_x_preprocessed.shape[1])])
test_x_preprocessed = pd.DataFrame(test_x_preprocessed,
                                   columns=[f'PC{i + 1}' for i in range(test_x_preprocessed.shape[1])])

# Phase 1: Match result prediction
result_cols = ['result_win_A', 'result_win_B', 'result_draw']
result_cols_in_x = [col for col in result_cols if col in train_x.columns]
decision_cols_in_x = [col for col in train_y.columns if col not in result_cols and col in train_x.columns]
x_result = train_x.drop(columns=result_cols_in_x + decision_cols_in_x)
y_result = train_y[result_cols]

y_result_combined = y_result.idxmax(axis=1)

x_train_result, x_test_result, y_train_result, y_test_result = train_test_split(
    x_result, y_result_combined, test_size=0.2, random_state=42, stratify=y_result_combined)

# Modello Random Forest con bilanciamento delle classi
rf_model_result = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf_model_result.fit(x_train_result, y_train_result)

y_pred_result = rf_model_result.predict(x_test_result)
print("Phase 1: Match Result Prediction")
print(classification_report(y_test_result, y_pred_result, zero_division=0))

save_model_zip(rf_model_result, 'rf_model_result.pkl')

# Phase 2: Decision type prediction for non-draw matches
decision_cols = [col for col in train_y.columns if col not in result_cols and col != 'decision_draw']
y_decision = train_y[decision_cols]

non_draw_indices_train = y_train_result != 'result_draw'
x_train_decision = x_train_result[non_draw_indices_train].copy()
y_train_decision = y_decision.loc[x_train_decision.index]

y_train_decision = y_train_decision.idxmax(axis=1).values.ravel()

non_draw_indices_test = y_test_result != 'result_draw'
x_test_decision = x_test_result[non_draw_indices_test].copy()
y_test_decision_filtered = y_decision.loc[x_test_decision.index]

y_test_decision_filtered = y_test_decision_filtered.idxmax(axis=1).values.ravel()

rf_model_decision = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')

if x_train_decision.shape[0] == y_train_decision.shape[0]:
    rf_model_decision.fit(x_train_decision, y_train_decision)
else:
    print("Dimension mismatch between x_train_decision and y_train_decision")

y_pred_decision = rf_model_decision.predict(x_test_decision)

print("Phase 2: Decision Type Prediction")
print(classification_report(y_test_decision_filtered, y_pred_decision, zero_division=0))

save_model_zip(rf_model_decision, 'rf_model_decision.pkl')
