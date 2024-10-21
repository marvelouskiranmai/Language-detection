import pandas as pd

# Read the data from Language Detection.csv
file_path = 'LanguageDetection.csv'
df = pd.read_csv(file_path)

# Remove null values
df_cleaned = df.dropna()

# Save the cleaned data to a new CSV file
output_path = 'cleaned_data.csv'
df_cleaned.to_csv(output_path, index=False)

print(f"Cleaned data saved to {output_path}")
