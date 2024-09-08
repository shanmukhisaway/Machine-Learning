#6

# Implementation of Random Forest Regressor (more effecoent than Decision Tree Regressor)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("CSV files/melb_data.csv")

y = df.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = df[melbourne_features]

# Split data into training and validation data, for both features and target
# The split is based on a random number generator
# Supplying a numeric value to the random_state argument guarantees we get the same split every time
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

forest_model = RandomForestRegressor(random_state = 1)
forest_model.fit(train_X, train_y)

melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))