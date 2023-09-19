import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create a synthetic dataset
data = {
    'Feature1': [1.2, 2.4, 3.1, 4.0, 5.2],
    'Feature2': [2.1, 3.5, 4.2, 5.1, 6.3],
    'Feature3': [0.5, 0.8, 1.0, 1.5, 2.0],
    'Sales': [10, 20, 30, 40, 50]
}

df = pd.DataFrame(data)

# Select features and target variable
X = df[['Feature1', 'Feature2', 'Feature3']]
y = df['Sales']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

# Make sales predictions for new scenarios
new_scenario = pd.DataFrame({'Feature1': [2.5], 'Feature2': [4.0], 'Feature3': [1.2]})
predicted_sales = model.predict(new_scenario)
print(f"Predicted Sales: {predicted_sales[0]:.2f}")

# Visualize the actual vs. predicted sales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, c='blue', label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k', c='red', label='Actual')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs. Predicted Sales")
plt.legend()
plt.show()
