from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load trained model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "house_price_model.pkl")
model = joblib.load(model_path)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form
    rm = float(data["rm"])
    lstat = float(data["lstat"])
    ptratio = float(data["ptratio"])

    input_data = np.array([[rm, lstat, ptratio]])
    prediction = model.predict(input_data)

    return render_template(
        "index.html",
        prediction_text=f"Predicted House Price: {prediction[0]:.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)
