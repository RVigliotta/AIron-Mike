import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.experimental import enable_iterative_imputer  # To enable IterativeImputer in scikit-learn
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.exceptions import ConvergenceWarning
import warnings


# Function to impute data using the specified imputer
def impute_data(imputer, data_, columns):
    data_copy = data_.copy()
    data_copy[columns] = imputer.fit_transform(data_copy[columns])
    return data_copy


# Function to compare correlations and calculate similarity percentage
def compare_correlations(original_data, imputed_data, columns, method_name):
    numeric_columns_ = original_data[columns].select_dtypes(include=[np.number]).columns
    original_corr = original_data[numeric_columns_].corr()
    imputed_corr = imputed_data[numeric_columns_].corr()
    similarity = np.corrcoef(original_corr.values.flatten(), imputed_corr.values.flatten())[0, 1] * 100
    print(f"Similarity percentage between correlations before and after imputation ({method_name}): {similarity:.2f}%")

    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    sns.heatmap(original_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlations of Original Data')
    plt.subplot(1, 2, 2)
    sns.heatmap(imputed_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlations After {method_name}')
    plt.tight_layout()
    plt.show()


# Function to handle outliers using Winsorization
def handle_outliers_winsorize(df, columns, lower_quantile=0.01, upper_quantile=0.99):
    df_cleaned = df.copy()
    for column in columns:
        lower_bound = df_cleaned[column].quantile(lower_quantile)
        upper_bound = df_cleaned[column].quantile(upper_quantile)
        df_cleaned[column] = np.clip(df_cleaned[column], lower_bound, upper_bound)
    return df_cleaned


# Ignore convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Loading the Dataset
data = pd.read_csv('../data/raw/boxing_matches.csv')
print("Null values per column")
print(data.isnull().sum().to_string())

# Count rows where `decision` is equal to 'NWS'
nws_count = data[data['decision'] == 'NWS'].shape[0]
print(f"Number of rows with result 'NWS': {nws_count}")

# Remove rows where `decision` is equal to 'NWS'
data = data[data['decision'] != 'NWS']

# Count rows where `result` is 'draw' and any decision is present
draw_decision_count = data[(data['result'] == 'draw') & (data['decision'].notnull())].shape[0]
print(f"Number of rows with result 'draw' and a decision: {draw_decision_count}")

# Set `decision` to an empty string where `result` is 'draw' and there is a decision
data.loc[(data['result'] == 'draw') & (data['decision'].notnull()), 'decision'] = 'draw'

# Stampa tutte le colonne del DataFrame
print(data.columns)

# Removing judge columns (discussed in documentation)
cols_to_drop = ['judge1_A', 'judge1_B', 'judge2_A', 'judge2_B', 'judge3_A', 'judge3_B']
data = data.drop(columns=cols_to_drop)

# Identifying important columns for imputation
important_cols = ['age_A', 'age_B', 'height_A', 'height_B', 'reach_A', 'reach_B', 'weight_A', 'weight_B']

# Removing rows with too many missing values in critical columns
data = data.dropna(subset=important_cols, thresh=len(important_cols) - 4)

# Save a copy of the dataset before imputation
data_before_imputation = data.copy()

# Imputation with MICE (using IterativeImputer from scikit-learn)
mice_imputer = IterativeImputer(max_iter=10, random_state=42, tol=1e-3)
data_mice = impute_data(mice_imputer, data, important_cols)

# Imputation with the most frequent value (Mode Imputation) for categorical variables
categorical_cols = ['stance_A', 'stance_B']
mode_imputer = SimpleImputer(strategy='most_frequent')
data_mice[categorical_cols] = mode_imputer.fit_transform(data_mice[categorical_cols])

# Fill the null value in `kos_B` with the median
data_mice['kos_B'] = data_mice['kos_B'].fillna(data_mice['kos_B'].median())

# Compare correlations and calculate similarity percentage after categorical imputation
compare_correlations(data_before_imputation, data_mice, important_cols + categorical_cols, '(MICE + Mode Imputation)')

# Count the null values per column
null_counts = data_mice.isnull().sum().to_string()
print("\nNull values per column after imputation:")
print(null_counts)
print("\nValues after imputation:")
print(data_mice.count().to_string())

print(f"Number of columns before One-Hot Encoding: {data_mice.shape[1]}")

# Check unique values in the 'result' column
print("Unique values in 'result' column before encoding:")
print(data_mice['result'].unique())

# Set `decision` to 'draw' where `result` is 'draw'
data_mice.loc[data_mice['result'] == 'draw', 'decision'] = 'draw'

# Apply One-Hot Encoding
data_encoded = pd.get_dummies(data_mice, columns=['stance_A', 'stance_B', 'result', 'decision'])
data_encoded = data_encoded.astype(int)

# Visualize the first few rows and the new columns after One-Hot Encoding
print(data_encoded.head())

# Check how many columns we now have after encoding
print(f"Number of columns after One-Hot Encoding: {data_encoded.shape[1]}")
print("New columns after encoding:")
print(data_encoded.columns)

# Columns to handle outliers
numeric_columns = ['age_A', 'age_B', 'height_A', 'height_B', 'reach_A', 'reach_B', 'weight_A', 'weight_B', 'won_A',
                   'won_B', 'lost_A', 'lost_B', 'drawn_A', 'drawn_B', 'kos_A', 'kos_B']

# Handle outliers using Winsorization
data_winsorized = handle_outliers_winsorize(data_encoded, numeric_columns, lower_quantile=0.01, upper_quantile=0.99)
print(f"Number of rows after handling outliers (Winsorization): {data_winsorized.shape[0]}")

# Select only the winsorized numeric columns
numeric_columns_winsorized = data_winsorized[numeric_columns]

# Plot the distribution of numerical variables after Winsorization
plt.figure(figsize=(16, 10))
sns.violinplot(data=numeric_columns_winsorized, inner="quartile", palette="muted")
plt.title('Distribution of Numerical Variables After Winsorization')
plt.xticks(rotation=45)
plt.show()

# Compare correlation matrices before and after Winsorization
compare_correlations(data_before_imputation, data_winsorized, important_cols, "Data Winsorized")

# Create new features
data_winsorized['experience_A'] = data_winsorized['won_A'] + data_winsorized['lost_A'] + data_winsorized['drawn_A']
data_winsorized['experience_B'] = data_winsorized['won_B'] + data_winsorized['lost_B'] + data_winsorized['drawn_B']
data_winsorized['win_percentage_A'] = data_winsorized['won_A'] / data_winsorized['experience_A']
data_winsorized['win_percentage_B'] = data_winsorized['won_B'] / data_winsorized['experience_B']
data_winsorized['ko_percentage_A'] = data_winsorized['kos_A'] / data_winsorized['won_A']
data_winsorized['ko_percentage_B'] = data_winsorized['kos_B'] / data_winsorized['won_B']

# Replace NaN values with 0
data_winsorized['experience_A'] = (data_winsorized['won_A'].fillna(0) + data_winsorized['lost_A'].fillna(0) +
                                   data_winsorized['drawn_A'].fillna(0))
data_winsorized['experience_B'] = (data_winsorized['won_B'].fillna(0) + data_winsorized['lost_B'].fillna(0) +
                                   data_winsorized['drawn_B'].fillna(0))
data_winsorized['win_percentage_A'] = data_winsorized['win_percentage_A'].fillna(0)
data_winsorized['win_percentage_B'] = data_winsorized['win_percentage_B'].fillna(0)
data_winsorized['ko_percentage_A'] = data_winsorized['ko_percentage_A'].fillna(0)
data_winsorized['ko_percentage_B'] = data_winsorized['ko_percentage_B'].fillna(0)

# Calculate the correlation matrix for the new features
new_features_corr = data_winsorized[
    ['experience_A', 'experience_B', 'win_percentage_A', 'win_percentage_B', 'ko_percentage_A',
     'ko_percentage_B']].corr()

# Visualize the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(new_features_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of New Features')
plt.show()

# Save the final dataset to a CSV file
output_path = '../data/processed/boxing_matches_processed.csv'
data_winsorized.to_csv(output_path, index=False)

# Stampa tutte le colonne del DataFrame
print(data.columns)

# Check if the file was saved successfully
if os.path.exists(output_path):
    print("Data Preprocessing completed successfully and file saved!")
else:
    print("Error: File was not saved successfully.")
