import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import joblib

# ================================
# 1. Load Dataset
# ================================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "housing.csv")

house_price_data = pd.read_csv(data_path)

# ================================
# 2. Exploratory Data Analysis
# ================================
correlation = house_price_data.corr()

plt.figure(figsize=(6, 6))
sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    cbar=True
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# ================================
# 3. Feature & Target Split
# ================================
X = house_price_data.drop("MEDV", axis=1)
y = house_price_data["MEDV"]

# ================================
# 4. Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# ================================
# 5. Hyperparameter Tuning (XGBoost)
# ================================
xgb = XGBRegressor(
    objective="reg:squarederror",
    random_state=42
)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0]
}

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=15,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    random_state=42,
    verbose=1
)

search.fit(X_train, y_train)

model = search.best_estimator_

print("Best Parameters:", search.best_params_)

# ================================
# 6. Model Evaluation
# ================================
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("\nModel Performance:")
print("Train R2:", r2_score(y_train, train_pred))
print("Test R2 :", r2_score(y_test, test_pred))
print("MAE     :", mean_absolute_error(y_test, test_pred))
print("RMSE    :", np.sqrt(mean_squared_error(y_test, test_pred)))

# ================================
# 7. Actual vs Predicted Plot
# ================================
plt.scatter(y_test, test_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# ================================
# 8. Save Model
# ================================
model_path = os.path.join(base_dir, "house_price_model.pkl")
joblib.dump(model, model_path)

print("\nModel saved as house_price_model.pkl")

# ================================
# 9. Sample Prediction
# ================================
sample_input = np.array([[6.575, 4.98, 15.3]])
sample_prediction = model.predict(sample_input)

print("\nSample Prediction:", sample_prediction[0])
