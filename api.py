import os

import joblib
import pickle
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5000", "http://127.0.0.1:5000",
    "http://localhost:5500", "http://127.0.0.1:5500",
    "https://localhost:443", "https://127.0.0.1:443",
    "http://localhost:63342",
    "null",
]}})

MODEL_DIR = "./model"


regression_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "randomforest_regressor_log1p_monthly_rent.joblib")
)

classification_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "svc_classifier_for_employee_attrition.joblib")
)
with open(
    os.path.join(MODEL_DIR, "association_rule_model.pkl"),
    "rb"
) as file:
    arm_artifact = pickle.load(file)
    recommendation_lookup = arm_artifact["recommendation_lookup"]
    lookup_table = arm_artifact["lookup_table"]

valid_names = set(lookup_table["item_name"])
name_to_id = dict(zip(lookup_table["item_name"], lookup_table["track_id"]))   # only used to label the output


def error(message, status=400):
    return jsonify({"error": message}), status

@app.route("/api/v1/models/rent-prediction-regressor/predictions", methods=["POST"])
def predict_rent():
    payload = request.get_json()

    input_row = pd.DataFrame([payload])

    predicted_log_rent = regression_pipeline.predict(input_row)[0]
    import numpy as np
    predicted_rent = float(np.expm1(predicted_log_rent))

    return jsonify({
        "predicted_monthly_rent_kes": round(predicted_rent, 2),
        "model": "Random Forest",
    })

@app.route("/api/v1/models/employee-attrition-classifier/predictions", methods=["POST"])
def predict_employee_attrition():
    payload = request.get_json()
    input_row = pd.DataFrame([payload])

    predicted_class = classification_pipeline.predict(input_row)[0]
    predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
    class_labels = classification_pipeline.named_steps["model"].classes_

    return jsonify({
        "predicted_employee_attrition_category": str(predicted_class),
        "class_probabilities": {
            str(label): round(float(prob), 4)
            for label, prob in zip(class_labels, predicted_probabilities)
        },
        "model": "Support Vector Classifier (tuned, SMOTE-resampled at training only)",
    })

@app.route("/api/v1/models/music-recommender/predictions", methods=["POST"])
def recommend_music():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return error("Send a JSON object, e.g. {\"basket\": [\"Deep Sleep\"]}.")

    raw_basket = payload.get("basket")
    if not isinstance(raw_basket, list) or not raw_basket:
        return error("basket must be a non-empty list of track names, e.g. [\"Deep Sleep\"].")

    try:
        top_n = int(payload.get("top_n", 5))
        if top_n < 1:
            raise ValueError
    except (TypeError, ValueError):
        return error("top_n must be a positive whole number.")

    unknown = [t for t in raw_basket if t not in valid_names]
    if unknown:
        return error(f"Unknown track(s): {', '.join(map(str, unknown))}")
    basket = set(raw_basket)

    # A rule fires when ALL its antecedent items are in the basket. If several rules
    # recommend the same track, keep the one with the highest confidence.
    candidates = {}
    for antecedent, rule_list in recommendation_lookup.items():
        if antecedent.issubset(basket):
            for rule in rule_list:
                for item in rule["consequents"]:
                    if item not in basket and (item not in candidates or rule["confidence"] > candidates[item][0]):
                        candidates[item] = (rule["confidence"], rule["lift"], rule["support"], antecedent)

    ranked = sorted(candidates.items(), key=lambda kv: (kv[1][0], kv[1][1]), reverse=True)[:top_n]
    recommendations = [{
        "track_id": name_to_id[name],
        "product_name": name,
        "confidence": round(float(conf), 3),
        "lift": round(float(lift), 2),
        "support": round(float(sup), 4),
        "because_you_played": sorted(ante),
    } for name, (conf, lift, sup, ante) in ranked]

    return jsonify({
        "basket": sorted(basket),
        "recommendations": recommendations,
        "message": None if recommendations else "No association rule matches this basket.",
        "model": "Apriori association rules, cleaned + precomputed lookup",
    })


if __name__ == "__main__":
    app.run(debug=True)