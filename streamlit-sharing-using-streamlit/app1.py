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


# ============================================================
# RENT PREDICTION TAB
# ============================================================

with tab_rent:
    pass


# ============================================================
# EMPLOYEE ATTRITION TAB
# ============================================================

with tab_attrition:
    pass


# ============================================================
# STORE SEGMENTATION TAB
# ============================================================

with tab_stores:
    pass


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