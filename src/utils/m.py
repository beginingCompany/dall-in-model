import pandas as pd
from config.paths import PathConfig


file_path = "data/processed/BIGINING_dataset.csv"
df = pd.read_csv(file_path)

# Show number of missing per column
print("Missing values per column:")
print(df.isnull().sum())

# Show rows with any missing value
missing_rows = df[df.isnull().any(axis=1)]
print("\nRows with missing values:")
print(missing_rows)

# Save to CSV for manual review
missing_rows.to_csv("missing_rows.csv", index=False)
