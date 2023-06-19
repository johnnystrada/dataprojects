# This is a general template for a data analysis workflow

### IMPORT PACKAGES ###
import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt

### DATA SET EVALUATION ###

# Load the CSV file
csv_file_path = 'path_to_csv_file.csv'  # Replace with actual CSV file path
data = pd.read_csv(csv_file_path)

# Output the dataset size
dataset_size = data.shape
print(f"Dataset size: {dataset_size}")

# Output the columns and their data types
column_info = data.dtypes
print("\nColumn information:")
print(column_info)

# Output value counts for each column
for column in data.columns:
    print(f"\nValue counts for {column}:")
    value_counts = data[column].value_counts()
    print(value_counts)

# Output summary statistics
summary_stats = data.describe()
print("\nSummary statistics:")
print(summary_stats)

# Visualize missing values
msno.bar(data)

# Function to replace missing values with the most frequent value
def fill_missing_with_most_frequent(df):
    for column in df.columns:
        most_frequent_value = df[column].mode()[0]
        df[column].fillna(most_frequent_value, inplace=True)

# Call the function to replace missing values
fill_missing_with_most_frequent(data)

# Verify the changes
print(data.head())

# Function to count unique values and store top value counts
def count_unique_values(df):
    topcolumn_counts = pd.DataFrame(columns=['Column', 'Value', 'Count'])
    
    for column in df.columns:
        value_counts = df[column].value_counts().reset_index()
        value_counts.columns = ['Value', 'Count']
        value_counts['Column'] = column
        top_values = value_counts.head()
        topcolumn_counts = pd.concat([topcolumn_counts, top_values], ignore_index=True)
    
    return topcolumn_counts


# Count unique values and store top value counts
topcolumn_counts = count_unique_values(data)

# Create bar graph to display topcolumn_counts
for column in data.columns:
    column_data = topcolumn_counts[topcolumn_counts['Column'] == column]
    plt.figure()
    plt.bar(column_data['Value'], column_data['Count'])
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title(f'Top Value Counts for {column}')

plt.show()

