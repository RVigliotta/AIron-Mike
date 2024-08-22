import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the Dataset
data = pd.read_csv('../data/raw/boxing_matches.csv')

# Initial Data Exploration
print("First rows of the dataset:")
print(data.head())

print("\nDataset information:")
print(data.info())

print("\nDescriptive statistics of the dataset:")
print(data.describe())

# Visualizing distributions for numerical variables
numeric_data = data.select_dtypes(include=[np.number])
num_cols = numeric_data.columns
half = len(num_cols) // 2

numeric_data.iloc[:, :half].hist(bins=30, figsize=(20, 15))
plt.suptitle('Distributions of Numerical Variables (Part 1)')
plt.show()

numeric_data.iloc[:, half:].hist(bins=30, figsize=(20, 15))
plt.suptitle('Distributions of Numerical Variables (Part 2)')
plt.show()

# Visualizing distributions for categorical variables
categorical_data = data.select_dtypes(include=[object])
for col in categorical_data.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(y=col, data=data)
    plt.title(f'Distribution of Categorical Variable: {col}')
    plt.show()

# Correlation matrix (excluding non-numeric columns)
plt.figure(figsize=(16, 14))
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Checking and Managing Missing Values (Exploration)
print("\nMissing values per column:")
missing_values = data.isnull().sum()
print(missing_values)

# Visualizing the percentage of missing values per column
missing_percentage = (missing_values / len(data)) * 100
print("Percentage of missing values per column:")
print(missing_percentage)

# Heatmap to visualize the position of missing values
plt.figure(figsize=(12, 8))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Identifying and visualizing outliers using violin plots
plt.figure(figsize=(16, 10))
sns.violinplot(data=numeric_data, inner="quartile", palette="muted")
plt.title('Outlier Distribution in Numerical Variables')
plt.xticks(rotation=45)
plt.show()
