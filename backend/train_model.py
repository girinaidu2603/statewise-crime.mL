import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

df = pd.read_csv("data/crime_data.csv")

X = df[['Murder','Rape','Theft','Robbery','Cyber_Crime']]
y = df['Total_Crime']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/crime_model.pkl")

print("Model trained successfully")
