"""Explore fertilizer_data.csv — columns, types, distributions."""
import pandas as pd

df = pd.read_csv("data/fertilizer_data.csv")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nNull counts:\n{df.isnull().sum()}")
print(f"\n=== Numeric columns ===")
print(df.describe())
print(f"\n=== Categorical columns ===")
for col in df.select_dtypes(include="object").columns:
    unique = df[col].unique()
    print(f"\n{col} ({len(unique)} unique): {sorted(unique)}")
print(f"\n=== Target: Fertilizer Name ===")
print(df["Fertilizer Name"].value_counts())
