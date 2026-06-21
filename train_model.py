"""
Generate price_estimator_model.pkl from training data.
Run: python train_model.py
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

df = pd.read_csv("user_big2.csv")
df = df.drop(columns=["user_id"])

cat_cols = [
    "gender", "occupation", "smoking_status", "pet_status",
    "sleep_schedule", "roommate_gender_pref", "pref_smoking",
]
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

TARGET = "budget_max"
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE : {mae:.2f} EGP")
print(f"R2  : {r2:.4f}")
print(f"Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")

joblib.dump(model, "price_estimator_model.pkl")
print("Saved: price_estimator_model.pkl")
