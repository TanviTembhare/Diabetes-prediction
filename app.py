from flask import Flask, request, jsonify
import numpy as np
import joblib
import traceback
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return "Diabetes Prediction API Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        feature_order = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]

        values = [data.get(f) for f in feature_order]

        arr = np.array(values).reshape(1, -1)
        arr_scaled = scaler.transform(arr)

        prediction = model.predict(arr_scaled)[0]

        return jsonify({
            "prediction": int(prediction),
            "result": "Diabetic" if prediction == 1 else "Not Diabetic"
        })

    except Exception:
        return jsonify({"error": traceback.format_exc()})


if __name__ == "__main__":
    app.run(debug=True)
