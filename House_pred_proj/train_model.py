import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Simple dataset (Area vs Price)
X = np.array([[500], [1000], [1500], [2000], [2500]])
y = np.array([100000, 200000, 300000, 400000, 500000])

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model trained and saved.")
