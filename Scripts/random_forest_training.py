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
train_X = pd.read_csv('../data/splitted/X_train.csv')
train_y = pd.read_csv('../data/splitted/y_train.csv')
test_X = pd.read_csv('../data/splitted/X_test.csv')
test_y = pd.read_csv('../data/splitted/y_test.csv')
val_X = pd.read_csv('../data/splitted/X_val.csv')
val_y = pd.read_csv('../data/splitted/y_val.csv')

print(f"Initial train_X dimensions: {train_X.shape}")
print(f"Initial train_y dimensions: {train_y.shape}")

train_X = replace_inf_and_nan(train_X)
val_X = replace_inf_and_nan(val_X)
test_X = replace_inf_and_nan(test_X)

# Outlier analysis
iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(train_X)
mask = yhat != -1
train_X_no_outliers = train_X[mask]
train_y_no_outliers = train_y[mask]

# Verify dimensions
print(f"train_X dimensions without outliers: {train_X_no_outliers.shape}")
print(f"train_y dimensions without outliers: {train_y_no_outliers.shape}")

# Ensure train_y_no_outliers is a 1D array
if train_y_no_outliers.ndim > 1:
    train_y_no_outliers = train_y_no_outliers.iloc[:, 0].values.ravel()

# Apply SMOTE
smote = SMOTE(random_state=42)
train_X_resampled, train_y_resampled = smote.fit_resample(train_X_no_outliers, train_y_no_outliers)

# Preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95))
])

train_X_preprocessed = preprocessing_pipeline.fit_transform(train_X_resampled)
val_X_preprocessed = preprocessing_pipeline.transform(val_X)
test_X_preprocessed = preprocessing_pipeline.transform(test_X)

# Convert to DataFrame
train_X_preprocessed = pd.DataFrame(train_X_preprocessed,
                                    columns=[f'PC{i + 1}' for i in range(train_X_preprocessed.shape[1])])
val_X_preprocessed = pd.DataFrame(val_X_preprocessed,
                                  columns=[f'PC{i + 1}' for i in range(val_X_preprocessed.shape[1])])
test_X_preprocessed = pd.DataFrame(test_X_preprocessed,
                                   columns=[f'PC{i + 1}' for i in range(test_X_preprocessed.shape[1])])

# Phase 1: Match result prediction
result_cols = ['result_win_A', 'result_win_B', 'result_draw']
print("Columns in train_X:", train_X.columns.tolist())
print("Columns in train_y:", train_y.columns.tolist())

# Ensure the columns exist in the DataFrame
result_cols_in_X = [col for col in result_cols if col in train_X.columns]
decision_cols_in_X = [col for col in train_y.columns if col not in result_cols and col in train_X.columns]

X_result = train_X.drop(columns=result_cols_in_X + decision_cols_in_X)
y_result = train_y[result_cols]

# Combine target columns into a single column for stratification
y_result_combined = y_result.idxmax(axis=1)

# Data split with stratification
X_train_result, X_test_result, y_train_result, y_test_result = train_test_split(X_result, y_result,
                                                                                test_size=0.2, random_state=42,
                                                                                stratify=y_result_combined)

# Model training
rf_model_result = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model_result.fit(X_train_result, y_train_result)

# Model evaluation for the first phase (result prediction)
y_pred_result = rf_model_result.predict(X_test_result)
print("Phase 1: Match Result Prediction")
print(classification_report(y_test_result, y_pred_result, zero_division=0))

# Phase 2: Decision type prediction for non-draw matches
decision_cols = [col for col in train_y.columns if col not in result_cols and col != 'decision_draw']
y_decision = train_y[decision_cols]

# Filter data for non-draw matches in the training set
non_draw_indices_train = y_train_result.idxmax(axis=1) != 'result_draw'
X_train_decision = X_train_result[non_draw_indices_train].copy()
y_train_decision = y_decision.loc[X_train_decision.index]

# Filter data for non-draw matches in the test set
non_draw_indices_test = y_test_result.idxmax(axis=1) != 'result_draw'
X_test_decision = X_test_result[non_draw_indices_test].copy()
y_test_decision_filtered = y_decision.loc[X_test_decision.index]

# Model training
rf_model_decision = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model_decision.fit(X_train_decision, y_train_decision)

# Decision type prediction
y_pred_decision = rf_model_decision.predict(X_test_decision)

# Convert y_pred_decision to DataFrame with the same columns as y_test_decision_filtered
y_pred_decision_df = pd.DataFrame(y_pred_decision, columns=y_test_decision_filtered.columns,
                                  index=y_test_decision_filtered.index)

# Model evaluation for the second phase (decision type)
print("Phase 2: Decision Type Prediction")
print(classification_report(y_test_decision_filtered, y_pred_decision_df, zero_division=0))
