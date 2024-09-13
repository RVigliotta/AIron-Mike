import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler, MaxAbsScaler


# Replace infinities and NaN with the column mean
def replace_inf_and_nan(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df


# Define a function to calculate the Euclidean distance using a for loop
def calculate_euclidean_distance(original_data, scaled_data):
    distances = []
    for i in range(len(original_data)):
        distances.append(euclidean(original_data[i], scaled_data.iloc[i]))
    return distances


# Load preprocessed and split datasets
train_X = pd.read_csv('../data/splitted/X_train.csv')
train_y = pd.read_csv('../data/splitted/y_train.csv')
test_X = pd.read_csv('../data/splitted/X_test.csv')
test_y = pd.read_csv('../data/splitted/y_test.csv')
val_X = pd.read_csv('../data/splitted/X_val.csv')
val_y = pd.read_csv('../data/splitted/y_val.csv')

train_X = replace_inf_and_nan(train_X)
val_X = replace_inf_and_nan(val_X)
test_X = replace_inf_and_nan(test_X)

# Outlier analysis
iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(train_X)
mask = yhat != -1
train_X_no_outliers = train_X[mask]

# Data normalization
scaler = StandardScaler()
train_X_normalized = scaler.fit_transform(train_X_no_outliers)
val_X_normalized = scaler.transform(val_X)
test_X_normalized = scaler.transform(test_X)

# Dimensionality reduction with PCA
pca = PCA(n_components=0.95)  # Keeps 95% of the variance
train_X_pca = pca.fit_transform(train_X_normalized)
val_X_pca = pca.transform(val_X_normalized)
test_X_pca = pca.transform(test_X_normalized)

# Apply MaxAbsScaler on features
scaler = (MaxAbsScaler())

train_X_scaled = scaler.fit_transform(train_X_pca)
val_X_scaled = scaler.transform(val_X_pca)
test_X_scaled = scaler.transform(test_X_pca)

# Convert to DataFrame
train_X_scaled = pd.DataFrame(train_X_scaled, columns=[f'PC{i + 1}' for i in range(train_X_scaled.shape[1])])
val_X_scaled = pd.DataFrame(val_X_scaled, columns=[f'PC{i + 1}' for i in range(val_X_scaled.shape[1])])
test_X_scaled = pd.DataFrame(test_X_scaled, columns=[f'PC{i + 1}' for i in range(test_X_scaled.shape[1])])

# Visualize the scaled data to verify the result
fig, axes = plt.subplots(3, 2, figsize=(15, 15))

sns.histplot(train_X, ax=axes[0, 0], kde=True, legend=False)
axes[0, 0].set_title('Original Distribution - Train')
sns.histplot(train_X_scaled, ax=axes[0, 1], kde=True, legend=False)
axes[0, 1].set_title('Scaled Distribution - Train')

sns.histplot(val_X, ax=axes[1, 0], kde=True, legend=False)
axes[1, 0].set_title('Original Distribution - Validation')
sns.histplot(val_X_scaled, ax=axes[1, 1], kde=True, legend=False)
axes[1, 1].set_title('Scaled Distribution - Validation')

sns.histplot(test_X, ax=axes[2, 0], kde=True, legend=False)
axes[2, 0].set_title('Original Distribution - Test')
sns.histplot(test_X_scaled, ax=axes[2, 1], kde=True, legend=False)
axes[2, 1].set_title('Scaled Distribution - Test')

plt.tight_layout()
plt.show()

# Evaluate similarity
original_vs_scaled_val = calculate_euclidean_distance(train_X_pca, train_X_scaled)
distance_val = np.mean(original_vs_scaled_val)
print(f"Mean Euclidean Distance - Train: {distance_val}")

original_vs_scaled_val = calculate_euclidean_distance(val_X_pca, val_X_scaled)
distance_val = np.mean(original_vs_scaled_val)
print(f"Mean Euclidean Distance - Validation: {distance_val}")

original_vs_scaled_test = calculate_euclidean_distance(test_X_pca, test_X_scaled)
distance_test = np.mean(original_vs_scaled_test)
print(f"Mean Euclidean Distance - Test: {distance_test}")
