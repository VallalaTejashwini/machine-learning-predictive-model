import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("student_scores.csv")

# Input and output
X = data[["Hours"]]
y = data["Marks"]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict values
predictions = model.predict(X)

# Accuracy score
score = r2_score(y, predictions)

print("Model Accuracy:")
print(score)

# Scatter plot
plt.scatter(data["Hours"], data["Marks"])

# Regression line
plt.plot(data["Hours"], predictions)

# Labels
plt.xlabel("Study Hours")
plt.ylabel("Marks")

# Title
plt.title("Machine Learning Prediction")

# Show graph
plt.show()