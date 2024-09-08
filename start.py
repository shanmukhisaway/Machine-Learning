#1

import pandas as pd

melbourne_file_path = "CSV files/melb_data.csv"

df = pd.read_csv(melbourne_file_path)

print(df.columns)
print(df.Price)

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = df[melbourne_features]
print(X.describe())