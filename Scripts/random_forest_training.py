import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Replace infinities and NaN with the column mean
def replace_inf_and_nan(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df


# Load preprocessed and split datasets
train_x = pd.read_csv('../data/splitted/x_train.csv')
train_y = pd.read_csv('../data/splitted/y_train.csv')
test_x = pd.read_csv('../data/splitted/x_test.csv')
test_y = pd.read_csv('../data/splitted/y_test.csv')
val_x = pd.read_csv('../data/splitted/x_val.csv')
val_y = pd.read_csv('../data/splitted/y_val.csv')

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

# Verify dimensions
print(f"train_x dimensions without outliers: {train_x_no_outliers.shape}")
print(f"train_y dimensions without outliers: {train_y_no_outliers.shape}")

# Ensure train_y_no_outliers is a 1D array
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

# Convert to DataFrame
train_x_preprocessed = pd.DataFrame(train_x_preprocessed,
                                    columns=[f'PC{i + 1}' for i in range(train_x_preprocessed.shape[1])])
val_x_preprocessed = pd.DataFrame(val_x_preprocessed,
                                  columns=[f'PC{i + 1}' for i in range(val_x_preprocessed.shape[1])])
test_x_preprocessed = pd.DataFrame(test_x_preprocessed,
                                   columns=[f'PC{i + 1}' for i in range(test_x_preprocessed.shape[1])])

# Phase 1: Match result prediction
result_cols = ['result_win_A', 'result_win_B', 'result_draw']
print("Columns in train_x:", train_x.columns.tolist())
print("Columns in train_y:", train_y.columns.tolist())

# Ensure the columns exist in the DataFrame
result_cols_in_x = [col for col in result_cols if col in train_x.columns]
decision_cols_in_x = [col for col in train_y.columns if col not in result_cols and col in train_x.columns]

x_result = train_x.drop(columns=result_cols_in_x + decision_cols_in_x)
y_result = train_y[result_cols]

# Combine target columns into a single column for stratification
y_result_combined = y_result.idxmax(axis=1)

# Data split with stratification
x_train_result, x_test_result, y_train_result, y_test_result = train_test_split(x_result, y_result,
                                                                                test_size=0.2, random_state=42,
                                                                                stratify=y_result_combined)

# Model training
rf_model_result = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model_result.fit(x_train_result, y_train_result)

# Model evaluation for the first phase (result prediction)
y_pred_result = rf_model_result.predict(x_test_result)
print("Phase 1: Match Result Prediction")
print(classification_report(y_test_result, y_pred_result, zero_division=0))

# Phase 2: Decision type prediction for non-draw matches
decision_cols = [col for col in train_y.columns if col not in result_cols and col != 'decision_draw']
y_decision = train_y[decision_cols]

# Filter data for non-draw matches in the training set
non_draw_indices_train = y_train_result.idxmax(axis=1) != 'result_draw'
x_train_decision = x_train_result[non_draw_indices_train].copy()
y_train_decision = y_decision.loc[x_train_decision.index]

# Filter data for non-draw matches in the test set
non_draw_indices_test = y_test_result.idxmax(axis=1) != 'result_draw'
x_test_decision = x_test_result[non_draw_indices_test].copy()
y_test_decision_filtered = y_decision.loc[x_test_decision.index]

# Model training
rf_model_decision = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model_decision.fit(x_train_decision, y_train_decision)

# Decision type prediction
y_pred_decision = rf_model_decision.predict(x_test_decision)

# Convert y_pred_decision to DataFrame with the same columns as y_test_decision_filtered
y_pred_decision_df = pd.DataFrame(y_pred_decision, columns=y_test_decision_filtered.columns,
                                  index=y_test_decision_filtered.index)

# Model evaluation for the second phase (decision type)
print("Phase 2: Decision Type Prediction")
print(classification_report(y_test_decision_filtered, y_pred_decision_df, zero_division=0))
