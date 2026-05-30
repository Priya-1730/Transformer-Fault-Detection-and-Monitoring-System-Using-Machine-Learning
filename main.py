from flask import Flask, request, jsonify
import pandas as pd
import os
from datetime import datetime
import joblib

# ---------------- INIT ----------------
app = Flask(__name__)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(BASE_DIR, "data.csv")
model_path = os.path.join(BASE_DIR, "model.pkl")

# ---------------- LOAD MODEL ----------------
model = joblib.load(model_path)

# ---------------- SAFE FLOAT ----------------
def safe_float(value):
    try:
        return float(value)
    except:
        return 0.0

# ---------------- HOME ----------------
@app.route('/')
def home():
    return "Flask Server Running ✅"

# ---------------- DATA API ----------------
@app.route('/data')
def data():

    try:

        # -------- RECEIVE ESP VALUES --------
        temp = safe_float(request.args.get('temp'))
        voltage = safe_float(request.args.get('voltage'))
        current = safe_float(request.args.get('current'))
        oil_distance = safe_float(request.args.get('oil'))

        # -------- OIL % --------
        tank_height = 30

        if oil_distance <= 0 or oil_distance > tank_height:
            oil_percent = 0
        else:
            oil_percent = ((tank_height - oil_distance) / tank_height) * 100

        oil_percent = max(0, min(100, oil_percent))

        # -------- ML PREDICTION --------
        input_df = pd.DataFrame(
            [[temp, voltage, current, oil_percent]],
            columns=[
                "Temperature",
                "Voltage",
                "Current",
                "Oil_Percentage"
            ]
        )

        prediction = model.predict(input_df)

        status = prediction[0]

        print(
            "DATA:",
            temp,
            voltage,
            current,
            oil_percent,
            "→",
            status
        )

        # -------- SAVE CSV --------
        new_data = pd.DataFrame([{
            "Time": datetime.now(),
            "Temperature": temp,
            "Voltage": voltage,
            "Current": current,
            "Oil_Percentage": oil_percent,
            "Status": status
        }])

        if os.path.exists(file_path):

            new_data.to_csv(
                file_path,
                mode='a',
                header=False,
                index=False
            )

        else:

            new_data.to_csv(
                file_path,
                index=False
            )

        return jsonify({
            "status": status
        })

    except Exception as e:

        print("ERROR:", e)

        return "Error", 500


# ---------------- RUN ----------------
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )