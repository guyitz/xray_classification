import pandas as pd

# Load the two CSV files
df1 = pd.read_csv('vit_chest_xray_prediction_no_finetune_newer_model_results.csv')
df2 = pd.read_csv('tiny_model_covid_classification_result.csv')

# Rename columns to clarify which model they belong to
df1 = df1.rename(columns={
    'classification': 'vit_model',
    'model_percentage': 'confidence_vit_model'
})

df2 = df2.rename(columns={
    'classification': 'classification_tiny_model',
    'model_percentage': 'confidence_tiny_model'
})

# Merge on image_name and directory (if both are needed to uniquely identify each image)
merged_df = pd.merge(df1, df2, on=['image_name', 'directory'], how='inner')

# Optional: Save to a new CSV
merged_df.to_csv('combined_model_results.csv', index=False)

# Display the result
print(merged_df.head())