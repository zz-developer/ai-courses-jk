import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic dataset
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # Independent variable
y = 4 + 3 * X + np.random.randn(100, 1)  # Dependent variable with some noise

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Visualize the dataset and regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", label="Data Points")
# plt.plot(X, 4 + 3 * X, color="green", linestyle="--", label="True Line")
plt.plot(X_test, y_pred, color="red", label="Regression Line")
plt.xlabel("X")
plt.ylabel("y")
plt.title(f"Simple Linear Regression, result function: y = {model.intercept_[0]:.2f} + {model.coef_[0][0]:.2f} * X")
plt.legend()
plt.grid()

# Metrics
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

plt.savefig("linear_regression.png")
