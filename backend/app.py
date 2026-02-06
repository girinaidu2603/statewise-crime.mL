from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load("model/crime_model.pkl")

@app.route("/")
def home():
    return {"status": "Statewise Crime ML API running"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = [[
        data["murder"],
        data["rape"],
        data["theft"],
        data["robbery"],
        data["cyber"]
    ]]
    prediction = model.predict(features)[0]
    return jsonify({"predicted_total_crime": int(prediction)})

if __name__ == "__main__":
    app.run()
