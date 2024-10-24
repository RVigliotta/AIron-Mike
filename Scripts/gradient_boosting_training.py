import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import Parallel, delayed


# Function to replace infinities and NaN with column mean
def replace_inf_and_nan(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df


# Function to load a CSV file
def load_csv(file_path):
    return pd.read_csv(file_path)


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

# Replace NaN and infinities
train_x = replace_inf_and_nan(train_x)
val_x = replace_inf_and_nan(val_x)
test_x = replace_inf_and_nan(test_x)

# Outlier analysis with Isolation Forest
iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(train_x)
mask = yhat != -1
train_x_no_outliers = train_x[mask]
train_y_no_outliers = train_y[mask]

print(f"train_x dimensions without outliers: {train_x_no_outliers.shape}")
print(f"train_y dimensions without outliers: {train_y_no_outliers.shape}")

# Ensure train_y_no_outliers is a 1D array
if train_y_no_outliers.ndim > 1:
    train_y_no_outliers = train_y_no_outliers.iloc[:, 0].values.ravel()

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
train_x_resampled, train_y_resampled = smote.fit_resample(train_x_no_outliers, train_y_no_outliers)

# Preprocessing pipeline with PCA
preprocessing_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95))
])

# Apply the pipeline to the data
train_x_preprocessed = preprocessing_pipeline.fit_transform(train_x_resampled)
val_x_preprocessed = preprocessing_pipeline.transform(val_x)
test_x_preprocessed = preprocessing_pipeline.transform(test_x)

# Convert to DataFrame for clarity
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

# Combine target columns into a single column for stratification
y_result_combined = y_result.idxmax(axis=1)

# Split data with stratification
x_train_result, x_test_result, y_train_result, y_test_result = train_test_split(
    x_result, y_result_combined, test_size=0.2, random_state=42, stratify=y_result_combined)

# Train Gradient Boosting model
gb_model_result = GradientBoostingClassifier(random_state=42, n_estimators=100)
gb_model_result.fit(x_train_result, y_train_result)

# Model evaluation (Phase 1: Match result prediction)
y_pred_result = gb_model_result.predict(x_test_result)
print("Phase 1: Match Result Prediction")
print(classification_report(y_test_result, y_pred_result, zero_division=0))

# Phase 2: Decision type prediction for non-draw matches
decision_cols = [col for col in train_y.columns if col not in result_cols and col != 'decision_draw']
y_decision = train_y[decision_cols]

# Filter data for non-draw matches in the training set
non_draw_indices_train = y_train_result != 'result_draw'
x_train_decision = x_train_result[non_draw_indices_train].copy()
y_train_decision = y_decision.loc[x_train_decision.index]

# Flatten y_train_decision
y_train_decision = y_train_decision.idxmax(axis=1).values.ravel()

# Filter data for non-draw matches in the test set
non_draw_indices_test = y_test_result != 'result_draw'
x_test_decision = x_test_result[non_draw_indices_test].copy()
y_test_decision_filtered = y_decision.loc[x_test_decision.index]

# Flatten y_test_decision_filtered
y_test_decision_filtered = y_test_decision_filtered.idxmax(axis=1).values.ravel()

# Train Gradient Boosting model for decision prediction
gb_model_decision = GradientBoostingClassifier(random_state=42, n_estimators=100)

# Ensure dimensions are equal before fitting the model
if x_train_decision.shape[0] == y_train_decision.shape[0]:
    gb_model_decision.fit(x_train_decision, y_train_decision)
else:
    print("Dimension mismatch between x_train_decision and y_train_decision")

# Decision prediction
y_pred_decision = gb_model_decision.predict(x_test_decision)

# Model evaluation (Phase 2: Decision type prediction)
print("Phase 2: Decision Type Prediction")
print(classification_report(y_test_decision_filtered, y_pred_decision, zero_division=0))
