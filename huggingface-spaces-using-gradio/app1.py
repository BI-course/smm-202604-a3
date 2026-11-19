import gradio as gr
import joblib
import numpy as np
import pandas as pd
import pickle
import spaces


@spaces.GPU
def _warm_zerogpu():
    return None


_warm_zerogpu()

regression_pipeline = joblib.load(
    "randomforest_regressor_log1p_monthly_rent.joblib"
)

classification_pipeline = joblib.load(
    "svc_classifier_for_employee_attrition.joblib"
)

with open("association_rule_model.pkl", "rb") as file:
    arm_artifact = pickle.load(file)

recommendation_lookup = arm_artifact["recommendation_lookup"]
lookup_table = arm_artifact["lookup_table"]

clustering_artifact = joblib.load("store_clustering_model.pkl")
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]


def jsonify(param):
    pass


def error(message, status=400):
    return jsonify({"error": message}), status


def predict_rent(property_age_years, size_sqm, num_bedrooms, num_bathrooms, distance_to_cbd_km, service_charge_kes,
                garage_size_sqm, security_rating, amenities_score, nearby_schools_count, property_type, neighborhood,
                furnishing_status):

    payload = {
        "property_age_years": property_age_years,
        "size_sqm": size_sqm,
        "num_bedrooms": num_bedrooms,
        "num_bathrooms": num_bathrooms,
        "distance_to_cbd_km": distance_to_cbd_km,
        "service_charge_kes": service_charge_kes,
        "garage_size_sqm": garage_size_sqm or None,
        "security_rating": security_rating or None,
        "amenities_score": amenities_score,
        "nearby_schools_count": nearby_schools_count,

        "property_type": property_type,
        "neighborhood": neighborhood,
        "furnishing_status": furnishing_status
    }

    input_row = pd.DataFrame([payload])
    predicted_log_rent = regression_pipeline.predict(input_row)[0]
    predicted_rent = float(np.expm1(predicted_log_rent))
    return f"KES {predicted_rent:,.2f} / month"


def predict_employee_attrition(department, job_role_level, education_level,
                               performance_rating, years_at_company,
                               years_since_last_promotion, monthly_salary_kes,
                               overtime_hours_per_month, job_satisfaction_score,
                               work_life_balance_score, distance_from_home_km,
                               num_companies_worked_previously,
                               remote_work_days_per_week, bonus_amount_kes):
    payload = {
        "department": department,
        "job_role_level": job_role_level,
        "education_level": education_level,
        "performance_rating": performance_rating,
        "years_at_company": years_at_company,
        "years_since_last_promotion": years_since_last_promotion,
        "monthly_salary_kes": monthly_salary_kes,
        "overtime_hours_per_month": overtime_hours_per_month,
        "job_satisfaction_score": job_satisfaction_score,
        "work_life_balance_score": work_life_balance_score,
        "distance_from_home_km": distance_from_home_km,
        "num_companies_worked_previously": num_companies_worked_previously,
        "remote_work_days_per_week": remote_work_days_per_week,
        "bonus_amount_kes": bonus_amount_kes or None,
    }

    input_row = pd.DataFrame([payload])

    predicted_class = classification_pipeline.predict(input_row)[0]
    predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]

    class_labels = classification_pipeline.named_steps["model"].classes_

    probability_lines = "\n".join(
        f"  {label}: {prob:.1%}"
        for label, prob in zip(class_labels, predicted_probabilities)
    )

    return f"Predicted_employee_attrition_category: {predicted_class}\n\n{probability_lines}"


def recommend_music(basket_text, top_n):
    basket = {track.strip() for track in basket_text.split(",") if track.strip()}

    candidates = {}

    for antecedent, rule_list in recommendation_lookup.items():
        if antecedent.issubset(basket):
            for rule in rule_list:
                for track in rule["consequents"]:
                    if track not in basket:
                        if track not in candidates or rule["confidence"] > candidates[track][0]:
                            candidates[track] = (
                                rule["confidence"],
                                rule["lift"],
                                rule["support"]
                            )

    ranked = sorted(
        candidates.items(),
        key=lambda item: (item[1][0], item[1][1]),
        reverse=True
    )[:int(top_n)]

    if not ranked:
        return "No recommendations for this basket -- try a combination of tracks seen in the training data."

    lines = [
        f"{track}  --  confidence {confidence:.1%}, lift {lift:.2f}, support {support:.4f}"
        for track, (confidence, lift, support) in ranked
    ]

    return "\n".join(lines)


def predict_store_segment(monthly_revenue_kes, daily_foot_traffic,
                          staff_count, store_size_sqm, avg_basket_size_kes):
    payload = {
        "monthly_revenue_kes": monthly_revenue_kes,
        "daily_foot_traffic": daily_foot_traffic,
        "staff_count": staff_count,
        "store_size_sqm": store_size_sqm,
        "avg_basket_size_kes": avg_basket_size_kes
    }

    feature_cols = clustering_artifact["features"]

    input_row = pd.DataFrame(
        [[payload.get(feature) for feature in feature_cols]],
        columns=feature_cols
    )

    scaled_values = clustering_scaler.transform(input_row)

    scaled_input = pd.DataFrame(
        scaled_values,
        columns=feature_cols
    )
    cluster_id = int(clustering_kmeans.predict(scaled_input)[0])

    assigned_cluster = cluster_id + 1
    cluster_names = ""

    if assigned_cluster == 1:
        cluster_names = "Smaller, lower-volume stores"
    elif assigned_cluster == 2:
        cluster_names = "Large, high-volume stores"
    elif assigned_cluster == 3:
        cluster_names = "High-traffic, low-basket stores"
    elif assigned_cluster == 4:
        cluster_names = "Low-traffic, high-basket stores"
    elif assigned_cluster == 5:
        cluster_names = "Mid-scale, near-average stores"

    return f"Assigned Segment: Cluster{cluster_names}"

with gr.Blocks(title="Group a3 Model Validation Demos") as demo:
    gr.Markdown(
        "# Group a3 – Model Validation Demos\n"
        "Switch between tabs to test the four machine-learning models."
    )

    # --------------------------------------------------
    # 1. RENTAL PRICE REGRESSOR
    with gr.Tab("Rental Price Regressor"):
        gr.Interface(
            fn=predict_rent,
            inputs=[
                gr.Number(label="Property Age (years)", value=5),
                gr.Number(label="Property Size (sqm)", value=120),
                gr.Number(label="Number of Bedrooms", value=3),
                gr.Number(label="Number of Bathrooms", value=2),
                gr.Number(label="Distance to CBD (km)", value=8),
                gr.Number(label="Service Charge (KES)", value=5000),
                gr.Number(label="Garage Size (sqm)", value=20),
                gr.Number(label="Security Rating", value=4),
                gr.Number(label="Amenities Score", value=8),
                gr.Number(label="Nearby Schools Count", value=3),

                gr.Dropdown(
                    label="Property Type",
                    choices=[
                        "Apartment",
                        "Townhouse",
                        "Masionette",
                        "Bungalow"
                    ],
                    value="Apartment"
                ),

                gr.Dropdown(
                    label="Neighbourhood",
                    choices=[
                        "Kilimani",
                        "Kileleshwa",
                        "Lavington",
                        "Karen",
                        "Westlands",
                        "Ngong Road",
                        "Syokimau",
                        "South B"
                    ],
                    value="Kilimani"
                ),

                gr.Dropdown(
                    label="Furnishing Status",
                    choices=[
                        "Unfurnished",
                        "Semi-Furnished",
                        "Fully Furnished"
                    ],
                    value="Semi-Furnished"
                )
            ],
            outputs=gr.Textbox(label="Predicted Monthly Rent"),
        )
    # --------------------------------------------------

    # --------------------------------------------------
    # 2. EMPLOYEE ATTRITION CLASSIFIER
    with gr.Tab("Employee Attrition Classifier"):
        gr.Interface(
            fn=predict_employee_attrition,
            inputs=[
                gr.Dropdown(
                    label="Department",
                    choices=[
                        "Operations",
                        "Finance",
                        "HR",
                        "Engineering",
                        "Sales"
                    ],
                    value="Engineering"
                ),

                gr.Dropdown(
                    label="Job Role Level",
                    choices=[
                        "Junior",
                        "Mid",
                        "Senior",
                        "Lead"
                    ],
                    value="Mid"
                ),

                gr.Dropdown(
                    label="Education Level",
                    choices=[
                        "Diploma",
                        "Bachelor's",
                        "Master's",
                        "PHD"
                    ],
                    value="Bachelor's"
                ),

                gr.Dropdown(
                    label="Performance Rating",
                    choices=[
                        "Low",
                        "Average",
                        "High",
                        "Outstanding"
                    ],
                    value="High"
                ),

                gr.Number(label="Years at Company", value=4),
                gr.Number(label="Years Since Last Promotion", value=2),
                gr.Number(label="Monthly Salary (KES)", value=85000),
                gr.Number(label="Overtime Hours per Month", value=15),
                gr.Number(label="Job Satisfaction Score", value=3),
                gr.Number(label="Work-Life Balance Score", value=3),
                gr.Number(label="Distance from Home (km)", value=12),
                gr.Number(label="Number of Previous Companies", value=2),
                gr.Number(label="Remote Work Days per Week", value=2),
                gr.Number(label="Bonus Amount (KES)", value=15000)
            ],
            outputs=gr.Textbox(
                label="Predicted Employee Attrition Category"
            ),
        )
    # --------------------------------------------------

    # --------------------------------------------------
    # 3. MUSIC RECOMMENDER
    # --------------------------------------------------
    with gr.Tab("Music Recommender"):

        gr.Interface(
            fn=recommend_music,
            inputs=[
                gr.Textbox(
                    label="Basket (comma-separated track names)",
                    value="Deep Sleep",
                    placeholder="e.g. Deep Sleep, Morning Energy"
                ),

                gr.Number(
                    label="Number of Recommendations",
                    value=5,
                    precision=0
                )
            ],
            outputs=gr.Textbox(
                label="Recommended Music Tracks"
            ),
        )

    # --------------------------------------------------
    # 4. STORE CUSTOMER SEGMENTER
    # --------------------------------------------------

if __name__ == "__main__":
    demo.launch()