import pandas as pd
import matplotlib.pyplot as plt

# Define column names (from UCI description)
column_names = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", 
    "PTRATIO", "B", "LSTAT", "MEDV"
]

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
data = pd.read_csv(url, delim_whitespace=True, header=None, names=column_names)


data["PPR"] = data["MEDV"] / data["RM"]

# Add a synthetic time feature (use row index as time)
data["Time"] = range(1, len(data) + 1)

# Plot Price per p vs Time
plt.figure(figsize=(10, 6))
plt.scatter(data["DIS"], data["MEDV"], marker="o", color="blue")
plt.xlabel("Time (synthetic index)")
plt.ylabel("Price per Room")
plt.title("Price per Unit Area vs. Time")
plt.legend()
plt.grid(True)
plt.show()