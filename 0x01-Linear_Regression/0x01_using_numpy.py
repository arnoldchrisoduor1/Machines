import numpy as np

# Model parameters (theta)
theta = np.array([50, 0.25, 30])

# Features for the id, with 1 for bias term.
x_house1 = np.array([1, 800, 2])

# Calculating the prediction using the dot product.
y_predicted_house1 = np.dot(theta, x_house1)

print(y_predicted_house1)