import pandas as pd


file_path = "data/raw/E0_2526.csv"

df = pd.read_csv(file_path)

print("\nNumber of rows:")
print(len(df))

print("\nNumber of columns:")
print(len(df.columns))

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)