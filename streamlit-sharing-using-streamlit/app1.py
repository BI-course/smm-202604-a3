import os
import requests
import streamlit as st

# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Group ML Models",
    page_icon="📊",
    layout="wide",
)

DEFAULT_API_URL = "http://127.0.0.1:5000"

API_BASE = st.sidebar.text_input(
    "Flask API URL",
    value=os.getenv("API_BASE_URL", DEFAULT_API_URL),
    help="Base URL where the Flask API is running.",
).strip().rstrip("/")

st.title("Group Machine Learning Models")
st.caption("Streamlit frontend connected to the Flask API.")


# ============================================================
# 2. SHARED FUNCTIONS
# ============================================================

def post_prediction(endpoint: str, payload: dict):
    """Send prediction requests to the Flask API."""
    try:
        response = requests.post(
            f"{API_BASE}{endpoint}",
            json=payload,
            timeout=30,
        )
    except requests.RequestException as exc:
        st.error(f"Could not connect to the Flask API: {exc}")
        return None

    try:
        data = response.json()
    except ValueError:
        data = {"error": response.text or "Invalid API response."}

    if not response.ok:
        st.error(data.get("error", f"API error: {response.status_code}"))
        return None

    return data


# ============================================================
# 3. APPLICATION TABS
# ============================================================

tab_rent, tab_attrition, tab_stores, tab_music = st.tabs(
    [
        "Rent Prediction",
        "Employee Attrition",
        "Store Segmentation",
        "Music Recommendation",
    ]
)


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
#MUSIC RECOMMENDATION TAB
# ============================================================

with tab_music:

       pass


# ============================================================
# 8. APPLICATION FOOTER
# ============================================================

st.divider()

st.caption(
    "Use model feature names and category values that match "
    "the training pipelines exactly."
)