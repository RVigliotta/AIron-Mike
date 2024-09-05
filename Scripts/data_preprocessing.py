import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer  # To enable IterativeImputer in scikit-learn
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.exceptions import ConvergenceWarning
import warnings

# Ignore convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Loading the Dataset
data = pd.read_csv('../data/raw/boxing_matches.csv')
print("Null values per column")
print(data.isnull().sum().to_string())  # Convert the Series to a string to avoid dtype output

# Count rows where `decision` is equal to 'NWS'
nws_count = data[data['decision'] == 'NWS'].shape[0]
print(f"Number of rows with result 'NWS': {nws_count}")

# Remove rows where `decision` is equal to 'NWS'
data = data[data['decision'] != 'NWS']

# Count rows where `result` is 'draw' and any decision is present
draw_decision_count = data[(data['result'] == 'draw') & (data['decision'].notnull())].shape[0]
print(f"Number of rows with result 'draw' and a decision: {draw_decision_count}")

# Remove rows where `result` is 'draw' and any decision is present
data = data[~((data['result'] == 'draw') & (data['decision'].notnull()))]

# Removing judge columns (discussed in documentation)
cols_to_drop = ['judge1_A', 'judge1_B', 'judge2_A', 'judge2_B', 'judge3_A', 'judge3_B']
data = data.drop(columns=cols_to_drop)

# Identifying important columns for imputation
important_cols = ['age_A', 'age_B', 'height_A', 'height_B', 'reach_A', 'reach_B', 'weight_A', 'weight_B']

# Removing rows with too many missing values in critical columns
data = data.dropna(subset=important_cols, thresh=len(important_cols) - 4)

# Save a copy of the dataset before imputation
data_before_imputation = data.copy()


# Function for data imputation
def impute_data(imputer, data, columns):
    data_copy = data.copy()
    data_copy[columns] = imputer.fit_transform(data_copy[columns])
    return data_copy


# Imputation with MICE (using IterativeImputer from scikit-learn)
mice_imputer = IterativeImputer(max_iter=10, random_state=42, tol=1e-3)
data_mice = impute_data(mice_imputer, data, important_cols)

# Imputation with the most frequent value (Mode Imputation) for categorical variables
categorical_cols = ['stance_A', 'stance_B']
mode_imputer = SimpleImputer(strategy='most_frequent')
data_mice[categorical_cols] = mode_imputer.fit_transform(data_mice[categorical_cols])

# Fill the null value in `kos_B` with the median
data_mice['kos_B'] = data_mice['kos_B'].fillna(data_mice['kos_B'].median())


# Function to compare correlations and calculate similarity percentage
def compare_correlations(original_data, imputed_data, columns, method_name):
    # Filter only numeric columns for correlation calculation
    numeric_columns = original_data[columns].select_dtypes(include=[np.number]).columns
    original_corr = original_data[numeric_columns].corr()
    imputed_corr = imputed_data[numeric_columns].corr()

    # Calculate the similarity percentage
    similarity = np.corrcoef(original_corr.values.flatten(), imputed_corr.values.flatten())[0, 1] * 100
    print(f"Similarity percentage between correlations before and after imputation ({method_name}): {similarity:.2f}%")

    plt.figure(figsize=(14, 7))

    # Correlations before imputation
    plt.subplot(1, 2, 1)
    sns.heatmap(original_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlations Before Imputation ({method_name})')

    # Correlations after imputation
    plt.subplot(1, 2, 2)
    sns.heatmap(imputed_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlations After Imputation ({method_name})')

    plt.tight_layout()
    plt.show()


# Compare correlations and calculate similarity percentage after categorical imputation
compare_correlations(data_before_imputation, data_mice, important_cols + categorical_cols, 'MICE + Mode Imputation')

# Count the null values per column
null_counts = data_mice.isnull().sum().to_string()
print("\nNull values per column after imputation:")
print(null_counts)
print("\nValues after imputation:")
print(data_mice.count().to_string())
