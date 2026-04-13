import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from xgboost import XGBRegressor

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("data/energy_data.csv", parse_dates=["datetime"])
df.set_index("datetime", inplace=True)

# -------------------------------
# 2. FEATURE ENGINEERING
# -------------------------------
df["hour"] = df.index.hour
df["day"] = df.index.day
df["month"] = df.index.month
df["dayofweek"] = df.index.dayofweek

# Lag features (IMPORTANT)
df["lag1"] = df["energy"].shift(1)
df["lag2"] = df["energy"].shift(2)
df["lag24"] = df["energy"].shift(24)

df.dropna(inplace=True)

# -------------------------------
# 3. DEFINE FEATURES
# -------------------------------
X = df.drop("energy", axis=1)
y = df["energy"]

# -------------------------------
# 4. TRAIN TEST SPLIT (TIME SERIES)
# -------------------------------
split = int(len(df) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -------------------------------
# 5. SCALING
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 6. MODEL (XGBOOST)
# -------------------------------
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8
)

model.fit(X_train, y_train)

# -------------------------------
# 7. PREDICTION
# -------------------------------
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ RMSE: {rmse}")
print(f"✅ R2 Score: {r2}")

# -------------------------------
# 8. PLOT RESULTS
# -------------------------------
plt.figure(figsize=(12,6))
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("Energy Forecasting")
plt.savefig("images/output.png")
plt.close()

print("📊 Graph saved: images/output.png")

# -------------------------------
# 9. FUTURE FORECAST (NEXT 24 HOURS)
# -------------------------------
future_steps = 24
last_row = df.iloc[-1:].copy()

future_preds = []

for i in range(future_steps):
    features = last_row.drop("energy", axis=1)

    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]

    future_preds.append(pred)

    # Update lag values
    last_row["lag2"] = last_row["lag1"]
    last_row["lag1"] = pred

# Save predictions
future_df = pd.DataFrame(future_preds, columns=["prediction"])
future_df.to_csv("outputs/future_predictions.csv", index=False)

print("🔮 Future predictions saved: outputs/future_predictions.csv")

# Plot future forecast
plt.figure(figsize=(10,5))
plt.plot(future_preds, marker="o")
plt.title("Future Energy Forecast")
plt.savefig("images/future_forecast.png")
plt.close()

print("📊 Future graph saved: images/future_forecast.png")