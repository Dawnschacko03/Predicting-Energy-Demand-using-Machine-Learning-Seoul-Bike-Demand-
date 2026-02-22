import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns
# Load dataset
df = pd.read_csv("C:/Users/donsh/Downloads/seoul+bike+sharing+demand/SeoulBikeData.csv", encoding='latin1')

# View first rows
print(df.head())
# Rename columns (remove special characters for easier handling)
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.replace("°C", "C")
df.columns = df.columns.str.replace("%", "Percent")

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Extract useful time-based features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.weekday

# Drop original Date column
df.drop('Date', axis=1, inplace=True)

# Convert categorical columns to numeric
df = pd.get_dummies(df, drop_first=True)

# Check for missing values
print(df.isnull().sum())

X = df.drop('Rented_Bike_Count', axis=1)
y = df['Rented_Bike_Count']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

def evaluate_model(y_test, y_pred, model_name):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"\n📌 {model_name} Performance:")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")


evaluate_model(y_test, y_pred_lr, "Linear Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_xgb, "XGBoost")

# Feature importance from Random Forest
importances = rf.feature_importances_
feature_names = X.columns

feat_importance = pd.Series(importances, index=feature_names)
feat_importance.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_xgb, alpha=0.5)
plt.xlabel("Actual Bike Count")
plt.ylabel("Predicted Bike Count")
plt.title("Actual vs Predicted - XGBoost")

max_val = max(y_test.max(), y_pred_xgb.max())
min_val = min(y_test.min(), y_pred_xgb.min())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.show()