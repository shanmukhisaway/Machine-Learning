#2

# Predict prices of the first 5 houses after analysing their features

from sklearn.tree import DecisionTreeRegressor
import pandas as pd

df = pd.read_csv("CSV files/melb_data.csv")

y = df.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = df[melbourne_features]

# Define model
melbourne_model = DecisionTreeRegressor(random_state = 1)

# Fit model
melbourne_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))