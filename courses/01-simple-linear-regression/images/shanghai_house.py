import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('shanghai_house.csv')

plt.figure(figsize=(10, 6))
plt.scatter(data['时间'], data['商品房均价'], color='blue', label='Data Points')

X = data['时间'].values.reshape(-1, 1)
y = data['商品房均价'].values

model = LinearRegression()
model.fit(X, y)

plt.xlabel('Time (Year)')
plt.ylabel('Price per Unit Area')
plt.title('Price per Unit Area vs. Time')
plt.legend()
plt.grid()
plt.xlim(2000, 2020)
plt.savefig('shanghai_house.png')

plt.plot(X, model.predict(X), color='red', label='Regression Line')

plt.savefig('shanghai_house_with_line.png')

# Clear the plot

plt.clf()

# Plot the residuals

plt.figure(figsize=(10, 6))
plt.scatter(X, y - model.predict(X), color='blue', label='Residuals')
plt.xlabel('Time (Year)')
plt.ylabel('Residuals')
plt.title('Residuals vs. Time')
plt.legend()
plt.grid()
plt.xlim(2000, 2020)
plt.savefig('shanghai_house_residuals.png')