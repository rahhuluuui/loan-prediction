import streamlit as st
import pandas as pd
import joblib
import json
import os
from datetime import datetime


# ==================================
# PAGE CONFIGURATION
# ==================================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)


# ==================================
# LOAD TRAINED MODEL
# ==================================

MODEL_PATH = "artifacts/loan_model.pkl"
METRICS_PATH = "artifacts/metrics.json"
SCHEMA_PATH = "artifacts/schema.json"


if not all(os.path.exists(path) for path in [
    MODEL_PATH, METRICS_PATH, SCHEMA_PATH
]):
    st.error("Model files not found. Please train the model first.")
    st.code("python train_model.py")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


model = load_model()
metrics = load_json(METRICS_PATH)
schema = load_json(SCHEMA_PATH)


# ==================================
# SESSION STATE
# ==================================

if "history" not in st.session_state:
    st.session_state.history = []


# ==================================
# PAGE TITLE
# ==================================

st.title("🏦 Loan Approval Prediction")

st.write(
    "This is a Machine Learning based application "
    "that predicts whether a loan application is "
    "likely to be approved or rejected."
)

st.write("---")


# ==================================
# PROJECT INFORMATION
# ==================================

st.subheader("Project Information")

st.write("**Algorithm:** Decision Tree Classifier")

st.write("**Problem Type:** Classification")

st.write("**Dataset:** Kaggle Loan Prediction Dataset")

st.write(
    f"**Model Accuracy:** "
    f"{metrics['accuracy'] * 100:.2f}%"
)

st.write("---")


# ==================================
# APPLICANT DETAILS FORM
# ==================================

st.subheader("Enter Applicant Details")

st.write(
    "Fill in the following information "
    "to predict your loan status."
)

input_data = {}

with st.form("loan_form"):

    numeric_features = schema["numeric_features"]
    categorical_features = schema["categorical_features"]

    for feature in schema["features"]:

        # Remove CoapplicantIncome from the form
        if feature.lower().replace("_", "") == "coapplicantincome":

            input_data[feature] = float(
                schema["defaults"].get(feature, 0)
            )

            continue

        # Display user-friendly feature names
        label = feature.replace("_", " ")

        if feature.lower() == "loan_amount_term":
            label = "Loan Repayment Period (Months)"

        elif feature.lower() == "loanamount":
            label = "Loan Amount"

        elif feature.lower() == "applicantincome":
            label = "Applicant Monthly Income"

        elif feature.lower() == "credit_history":
            label = "Credit History"

        elif feature.lower() == "property_area":
            label = "Property Area"

        elif feature.lower() == "self_employed":
            label = "Self Employed"

        elif feature.lower() == "dependents":
            label = "Number of Dependents"

        elif feature.lower() == "education":
            label = "Education"

        elif feature.lower() == "married":
            label = "Marital Status"

        elif feature.lower() == "gender":
            label = "Gender"

        # Numeric input fields
        if feature in numeric_features:

            default = float(
                schema["defaults"].get(feature, 0)
            )

            if feature.lower() == "credit_history":

                input_data[feature] = st.selectbox(
                    label,
                    options=[1.0, 0.0],
                    format_func=lambda x:
                        "Good (1)" if x == 1.0
                        else "Poor (0)"
                )

            else:

                input_data[feature] = st.number_input(
                    label,
                    min_value=0.0,
                    value=max(0.0, default),
                    step=1.0
                )

        # Categorical input fields
        else:

            options = schema["categories"].get(
                feature, []
            )

            options = [str(x) for x in options]

            default = str(
                schema["defaults"].get(
                    feature,
                    options[0] if options else "Unknown"
                )
            )

            if default not in options:
                options.insert(0, default)

            if not options:
                options = ["Unknown"]

            input_data[feature] = st.selectbox(
                label,
                options=options
            )

    st.write("")

    submitted = st.form_submit_button(
        "Predict Loan Status",
        use_container_width=True
    )


# ==================================
# LOAN PREDICTION
# ==================================

if submitted:

    input_df = pd.DataFrame(
        [input_data],
        columns=schema["features"]
    )

    # Convert numeric columns
    for col in numeric_features:
        input_df[col] = pd.to_numeric(
            input_df[col],
            errors="coerce"
        )

    prediction = int(
        model.predict(input_df)[0]
    )

    # Prediction result
    if prediction == 1:

        result = "Approved"

        st.success(
            "Congratulations! Your loan is predicted "
            "to be APPROVED."
        )

    else:

        result = "Rejected"

        st.error(
            "Sorry! Your loan is predicted "
            "to be REJECTED."
        )

    # Calculate model confidence
    probabilities = model.predict_proba(input_df)[0]

    class_index = list(
        model.classes_
    ).index(prediction)

    confidence = probabilities[class_index] * 100

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Loan Status",
            result
        )

    with col2:
        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

    st.caption(
        "Model confidence is an estimate from the Decision Tree. "
        "It is not a guarantee of actual bank approval."
    )

    # Save prediction history
    record = {
        "Date": datetime.now().strftime(
            "%d-%m-%Y %H:%M"
        ),
        **input_data,
        "Loan Status": result,
        "Confidence (%)": round(confidence, 2)
    }

    st.session_state.history.append(record)


# ==================================
# PREDICTION HISTORY
# ==================================

st.write("---")

st.subheader("Prediction History")

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    # Hide CoapplicantIncome from history too
    history_df = history_df.drop(
        columns=[
            col for col in history_df.columns
            if col.lower().replace("_", "") == "coapplicantincome"
        ],
        errors="ignore"
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    csv = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Prediction History",
        data=csv,
        file_name="loan_prediction_history.csv",
        mime="text/csv"
    )

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

else:

    st.info(
        "No predictions made yet. "
        "Enter applicant details to get started."
    )


# ==================================
# FOOTER
# ==================================

st.write("---")

st.caption(
    "Developed by Section 6B | Machine Learning Project"
)