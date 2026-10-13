from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("shopsmart_intent_classifier.joblib")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "project": "ShopSmart Retail - Customer Intent Classifier",
        "status": "API is running",
        "endpoint": "/predict"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({
            "error": "Please provide customer text."
        }), 400

    text = str(data["text"]).strip()

    if not text:
        return jsonify({
            "error": "Text cannot be empty."
        }), 400

    prediction = model.predict([text])[0]

    probabilities = model.predict_proba([text])[0]
    confidence = float(probabilities.max())

    return jsonify({
        "input": text,
        "predicted_intent": prediction,
        "confidence": round(confidence, 4)
    })


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)