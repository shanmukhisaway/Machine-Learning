#5

# Optimize the model by finding the sweet-spot between overfitting and underfitting

'''
Models can suffer from either:

Overfitting: capturing spurious patterns that won't recur in the future, leading to less accurate predictions, or
Underfitting: failing to capture relevant patterns, again leading to less accurate predictions.
'''

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)

    return mae

df = pd.read_csv("CSV files/melb_data.csv")

y = df.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = df[melbourne_features]

# Split data into training and validation data, for both features and target
# The split is based on a random number generator
# Supplying a numeric value to the random_state argument guarantees we get the same split every time
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

    print("Max leaf nodes:", max_leaf_nodes, "\t\t Mean Absolute Error:", my_mae)

# We can see that 500 is the optimal number of nodes