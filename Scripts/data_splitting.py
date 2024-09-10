import pandas as pd
from sklearn.model_selection import train_test_split
import os


# Function to validate file paths
def validate_file_path(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    if not os.path.isfile(file_path):
        raise ValueError(f"The path {file_path} is not a file.")


# Function to save datasets and check if the save was successful
def save_dataset(data, file_path):
    data.to_csv(file_path, index=False)
    if os.path.exists(file_path):
        print(f"File saved successfully: {file_path}")
    else:
        print(f"Error: File was not saved successfully: {file_path}")


# Load the preprocessed dataset
dataset_path = '../data/processed/boxing_matches_processed.csv'
validate_file_path(dataset_path)
df = pd.read_csv(dataset_path)

# Separate target columns (result and decision) from features
target_cols = ['result_win_A', 'result_win_B', 'result_draw', 'decision_DQ',
               'decision_KO', 'decision_MD', 'decision_PTS', 'decision_RTD',
               'decision_SD', 'decision_TD', 'decision_TKO', 'decision_UD', 'decision_draw']

# Check if target columns exist in the DataFrame
missing_cols = [col for col in target_cols if col not in df.columns]
if missing_cols:
    raise KeyError(f"Columns {missing_cols} not found in DataFrame")

X = df.drop(columns=target_cols)  # These are the features
y = df[target_cols]  # These are the targets

# Split the dataset into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the training set into training and validation sets (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Check the dimensions to verify the correct split
print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Validation set: {X_val.shape}, {y_val.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")

# Save the split datasets to a preprocessed folder
# Define the path to save the files (modify with your folder path)
output_folder = '../data/splitted/'

# Create the folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Save the datasets as CSV
save_dataset(X_train, os.path.join(output_folder, 'X_train.csv'))
save_dataset(y_train, os.path.join(output_folder, 'y_train.csv'))
save_dataset(X_val, os.path.join(output_folder, 'X_val.csv'))
save_dataset(y_val, os.path.join(output_folder, 'y_val.csv'))
save_dataset(X_test, os.path.join(output_folder, 'X_test.csv'))
save_dataset(y_test, os.path.join(output_folder, 'y_test.csv'))

print("Data saved successfully in the preprocessed folder.")
