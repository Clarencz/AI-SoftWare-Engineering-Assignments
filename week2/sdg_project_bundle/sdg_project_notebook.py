# SDG 13 - Forecasting CO2 emissions (simple supervised learning example)
# File: sdg_project_notebook.py
# This script demonstrates a full pipeline: load data, preprocess, train, evaluate, visualize.
# It uses a small synthetic dataset included in the bundle.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv("synthetic_co2_dataset.csv")
print("Dataset preview:")
print(df.head())

# 2. Feature engineering
# Use numerical features: year (as offset), gdp_per_capita, population, energy_consumption_per_capita
df["year_offset"] = df["year"] - df["year"].min()

features = ["year_offset", "gdp_per_capita", "population", "energy_consumption_per_capita"]
target = "co2_per_capita"

X = df[features]
y = df[target]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.3f}")
print(f"R2: {r2:.3f}")
print("Coefficients:", dict(zip(features, model.coef_)))
print("Intercept:", model.intercept_)

# 6. Visualization
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle="--")
plt.xlabel("True CO2 per capita")
plt.ylabel("Predicted CO2 per capita")
plt.title("True vs Predicted CO2 per capita")
plt.tight_layout()
plt.savefig("true_vs_predicted.png")
print("Saved plot to true_vs_predicted.png")

# 7. Simple future forecast example (forecast for 2025 for each country)
future_year = 2025
future_rows = []
for c in df["country"].unique():
    last = df[df["country"]==c].sort_values("year").iloc[-1]
    # naive projection: use last values for socio-economic features and update year_offset
    row = {
        "country": c,
        "year": future_year,
        "year_offset": future_year - df["year"].min(),
        "gdp_per_capita": last["gdp_per_capita"] * 1.05,  # assume 5% growth
        "population": last["population"] * 1.02,           # assume 2% growth
        "energy_consumption_per_capita": last["energy_consumption_per_capita"] * 1.02
    }
    future_rows.append(row)
future_df = pd.DataFrame(future_rows)
X_future = future_df[["year_offset","gdp_per_capita","population","energy_consumption_per_capita"]]
future_df["predicted_co2_per_capita"] = model.predict(X_future)
print("Future forecast (2025) sample:")
print(future_df[["country","year","predicted_co2_per_capita"]])
