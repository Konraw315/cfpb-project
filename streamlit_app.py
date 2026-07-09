""""
Streamlit front-end for the CFPB complaint response predictor.

Before running, the API needs to be running:
    uvicorn cfpb_api:app --reload

models/cfpb_predictor.joblib must be a dict with keys:
    model          : trained XGBClassifier
    encoder        : fitted LabelEncoder
    feature_dtypes : dict[column_name, training_dtype]

That file is created by running xgboost_optuna.py.
"""

import json

import matplotlib.pyplot as plt
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"
JSON_URL = ("https://cfpb-project-742435394310-us-east-1-an.s3.us-east-1.amazonaws.com/categories/categories.json")


st.set_page_config(
    page_title="CFPB Response Predictor",
    page_icon="📋",
    layout="centered",
)


@st.cache_data
def get_categories() -> dict[str, list[str]]:
    """Fetch dropdown categories once per session, not per rerun.

    Streamlit reruns this whole script on every widget interaction
    — without caching, every dropdown click would re-fetch this
    same JSON from S3 again.
    """
    response = requests.get(JSON_URL, timeout=10)
    response.raise_for_status()
    return json.loads(response.text)


st.title("📋 CFPB Response Predictor")
st.caption(
    "Predicts how a company is likely to respond to a consumer "
    "complaint, based on a model trained on CFPB complaint data."
)

try:
    categories = get_categories()
except requests.exceptions.RequestException as exc:
    st.error(f"Could not load dropdown options: {exc}")
    st.stop()

with st.form("complaint_form"):
    st.subheader("Enter the complaint details")

    col1, col2 = st.columns(2)
    columns = [col1, col2]
    input_dict: dict[str, str] = {}

    for i, (name, options) in enumerate(categories.items()):
        with columns[i % 2]:
            input_dict[name] = st.selectbox(name, options=options)

    submitted = st.form_submit_button("Predict", use_container_width=True)

with st.expander("Request payload"):
    st.json(input_dict)

if submitted:
    with st.spinner("Running model..."):
        try:
            response = requests.post(
                f"{API_URL}/predict", json=input_dict, timeout=30
            )
        except requests.exceptions.RequestException as exc:
            st.error(
                f"Could not reach the API at {API_URL}. Is "
                f"`uvicorn cfpb_api:app --reload` running? ({exc})"
            )
            st.stop()

    if response.ok:
        result = response.json()
        with st.container(border=True):
            st.subheader("Prediction")
            st.success(result["prediction"])
            st.metric(
                label="Model confidence",
                value=f"{result["confidence"]:.1%}",
            )

            st.progress(result["confidence"])

            st.caption(
            """Confidence represents the probability assigned by the model, 
            not a guarantee that the prediction is correct."""
            )
