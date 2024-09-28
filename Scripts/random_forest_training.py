import pandas as pd
import seaborn as sns
import numpy as np
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold


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

# Visualize the scaled data to verify the result
fig, axes = plt.subplots(3, 2, figsize=(15, 15))

sns.histplot(train_X, ax=axes[0, 0], kde=True, legend=False)
axes[0, 0].set_title('Original Distribution - Train')
sns.histplot(train_X_preprocessed, ax=axes[0, 1], kde=True, legend=False)
axes[0, 1].set_title('Preprocessed Distribution - Train')

sns.histplot(val_X, ax=axes[1, 0], kde=True, legend=False)
axes[1, 0].set_title('Original Distribution - Validation')
sns.histplot(val_X_preprocessed, ax=axes[1, 1], kde=True, legend=False)
axes[1, 1].set_title('Preprocessed Distribution - Validation')

sns.histplot(test_X, ax=axes[2, 0], kde=True, legend=False)
axes[2, 0].set_title('Original Distribution - Test')
sns.histplot(test_X_preprocessed, ax=axes[2, 1], kde=True, legend=False)
axes[2, 1].set_title('Preprocessed Distribution - Test')

plt.tight_layout()
plt.show()

# Define the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Configure k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_results_preprocessed = cross_val_score(rf_model, train_X_preprocessed, train_y_resampled, cv=kfold,
                                          scoring='accuracy')

# Print cross-validation results for the preprocessed model
print(f"Preprocessed Data")
print(f"Cross-Validation Scores: {cv_results_preprocessed}")
print(f"Mean Accuracy: {cv_results_preprocessed.mean()}")
print(f"Standard Deviation: {cv_results_preprocessed.std()}")
