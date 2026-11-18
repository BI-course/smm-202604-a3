import os
import pickle

import joblib
import requests
import streamlit as st


@st.cache_resource
def load_artifacts():
    MODEL_DIR = "./model"

    regression_pipeline = joblib.load(
        os.path.join(MODEL_DIR, "randomforest_regressor_log1p_monthly_rent.joblib")
    )
    classification_pipeline = joblib.load(
        os.path.join(MODEL_DIR, "svc_classifier_for_employee_attrition.joblib")
    )
    with open(os.path.join(MODEL_DIR, "association_rule_model.pkl"), "rb") as file:
        arm_artifact = pickle.load(file)
    clustering_artifact = joblib.load(
        os.path.join(MODEL_DIR, "store_clustering_model.pkl")
    )
    return (
        regression_pipeline,
        classification_pipeline,
        clustering_artifact,
        arm_artifact,
    )


regression_pipeline, classification_pipeline, clustering_artifact, arm_artifact = load_artifacts()

clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]
recommendation_lookup = arm_artifact["recommendation_lookup"]
lookup_table = arm_artifact["lookup_table"]

st.set_page_config(
    page_title="Group A3 Model Validation Demos",
    page_icon="\U0001F4CA",
    layout="centered",
)
st.title("Group A3 -- Model Validation Demos")

tab_rent, tab_attrition, tab_stores, tab_music = st.tabs([
    "Predict Rent",
    "Employee Attrition Classifier",
    "Store Segmenter",
    "Music Recommender",
])

DEFAULT_API_URL = "http://127.0.0.1:5000"

API_BASE = st.sidebar.text_input(
    "Flask API URL",
    value=DEFAULT_API_URL
).rstrip("/")

def post_prediction(endpoint, payload):
    try:
        response = requests.post(
            f"{API_BASE}{endpoint}",
            json=payload,
            timeout=30
        )

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as error:
        st.error(f"API request failed: {error}")
        return None

# ============================================================
# RENT PREDICTION TAB
# ============================================================

with tab_rent:
    st.header("Rental Price Prediction")
    st.write("Enter the property details to estimate its monthly rental price.")

    with st.form("rent_prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            neighborhood = st.selectbox(
                "Neighborhood",
                [
                    "Kilimani",
                    "Kileleshwa",
                    "Lavington",
                    "Karen",
                    "Westlands",
                    "Ngong Road",
                    "Syokimau",
                    "South B"
                ]
            )

            property_type = st.selectbox(
                "Property Type",
                [
                    "Apartment",
                    "Townhouse",
                    "Masionette",
                    "Bungalow"
                ]
            )

            furnishing_status = st.selectbox(
                "Furnishing",
                [
                    "Unfurnished",
                    "Semi-Furnished",
                    "Fully Furnished"
                ]
            )

            size_sqm = st.number_input(
                "Property Size (sqm)", min_value=1.0, value=50.0
            )

        with c2:
            num_bedrooms = st.number_input(
                "Number of Bedrooms", min_value=0, value=1, step=1
            )

            num_bathrooms = st.number_input(
                "Number of Bathrooms", min_value=0, value=1, step=1
            )

            distance_to_cbd_km = st.number_input(
                "Distance to CBD (km)", min_value=0.0, value=5.0
            )

            property_age_years = st.number_input(
                "Property Age (years)", min_value=0, value=5, step=1
            )

        with c3:
            service_charge_kes = st.number_input(
                "Service Charge (KES)", min_value=0.0, value=0.0
            )

            amenities_score = st.number_input(
                "Amenities Score", min_value=0.0, value=5.0
            )

            nearby_schools_count = st.number_input(
                "Nearby Schools", min_value=0, value=2, step=1
            )

            garage_size_sqm = st.number_input(
                "Garage Size (sqm)", min_value=0.0, value=0.0
            )

            security_rating = st.number_input(
                "Security Rating", min_value=0.0, value=5.0
            )

        submitted = st.form_submit_button(
            "Predict Rental Price",
            use_container_width=True
        )

    if submitted:
        payload = {
            "property_type": property_type,
            "neighborhood": neighborhood,
            "furnishing_status": furnishing_status,
            "size_sqm": size_sqm,
            "num_bedrooms": num_bedrooms,
            "num_bathrooms": num_bathrooms,
            "distance_to_cbd_km": distance_to_cbd_km,
            "property_age_years": property_age_years,
            "service_charge_kes": service_charge_kes,
            "amenities_score": amenities_score,
            "nearby_schools_count": nearby_schools_count,
            "garage_size_sqm": garage_size_sqm,
            "security_rating": security_rating
        }

        data = post_prediction(
            "/api/v1/models/rent-prediction-regressor/predictions",
            payload
        )

        if data:
            st.success("Rental price prediction completed!")

            if "predicted_monthly_rent_kes" in data:
                st.metric(
                    "Predicted Monthly Rent",
                    money_kes(data["predicted_monthly_rent_kes"])
                )

            with st.expander("View API Response"):
                st.json(data)



# ============================================================
# EMPLOYEE ATTRITION TAB
# ============================================================

with tab_attrition:
    pass


# ============================================================
# STORE SEGMENTATION TAB
# ============================================================

with tab_stores:
    with tab_stores:
        st.header("Store Segmentation")
        st.write(
            "Enter the store's operational and financial details "
            "to determine its segment."
        )

        with st.form("store_segmentation_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                monthly_revenue_kes = st.number_input(
                    "Monthly Revenue (KES)",
                    value=0.0
                )

                daily_foot_traffic = st.number_input(
                    "Daily Foot Traffic",
                    value=0,
                    step=1
                )

            with c2:
                staff_count = st.number_input(
                    "Number of Staff",
                    value=0,
                    step=1
                )

                store_size_sqm = st.number_input(
                    "Store Size (Square Metres)",
                    value=0.0
                )

            with c3:
                avg_basket_size_kes = st.number_input(
                    "Average Basket Size (KES)",
                    value=0.0
                )

            submitted = st.form_submit_button(
                "Predict Store Segment",
                use_container_width=True
            )

        if submitted:
            payload = {
                "monthly_revenue_kes": monthly_revenue_kes,
                "daily_foot_traffic": daily_foot_traffic,
                "staff_count": staff_count,
                "store_size_sqm": store_size_sqm,
                "avg_basket_size_kes": avg_basket_size_kes
            }

            if any(value <= 0 for value in payload.values()):
                st.warning("Please enter values greater than zero.")
            else:
                data = post_prediction(
                    "/api/v1/models/kmeans-store-segmentation/predictions",
                    payload
                )

                if data:
                    st.success("Store segmentation completed!")

                    st.metric(
                        "Assigned Cluster",
                        data["assigned_cluster"]
                    )

                    st.subheader("Store Segment")
                    st.write(data["segment_name"])

                    with st.expander("View API Response"):
                        st.json(data)

# ============================================================
# MUSIC RECOMMENDATION TAB
# ============================================================

with tab_music:
    pass


# ============================================================
# APPLICATION FOOTER
# ============================================================

st.divider()

st.caption(
    "Use model feature names and category values that match "
    "the training pipelines exactly."
)