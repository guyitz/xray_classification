import pandas as pd

# Load the CSV files
df1 = pd.read_csv("model_1_results.csv")
df2 = pd.read_csv("model_2_results.csv")

# Rename columns to avoid conflicts
df1 = df1.rename(columns={
    'classification': 'model_1_classification',
    'model_percentage': 'model_1_percentage'
})

df2 = df2.rename(columns={
    'classification': 'model_2_classification',
    'model_percentage': 'model_2_percentage'
})

# Merge on image_name and directory (if directory is needed)
merged_df = pd.merge(df1, df2, on=['image_name', 'directory'])

# Save the combined results
merged_df.to_csv("combined_model_results.csv", index=False)

print(merged_df.head())