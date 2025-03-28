import streamlit as st
import joblib

# --- Custom Modern Styling ---
st.markdown(
    """
    <style>
    body {
        background-color: #0e1117;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        font-size: 1.1em;
        font-weight: 600;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff7373;
        transform: scale(1.02);
    }
    .stAlert p {
        color: white !important;
        font-size: 1.1em;
    }
    .stMetric {
        background-color: #1e1e2f;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.9em;
        color: #999;
        margin-top: -10px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- Load Model ---
model = joblib.load("xgboost_model.pkl")

# --- Page Title ---
st.markdown(
    """
    <h1 style='color: #ffcc00;'>💰 Bank Term Deposit Subscription Predictor</h1>
""",
    unsafe_allow_html=True,
)
st.caption("Predict whether a client will subscribe based on their profile.")

# --- Column Layout ---
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox(
        "Job",
        [
            "admin.",
            "blue-collar",
            "entrepreneur",
            "housemaid",
            "management",
            "retired",
            "self-employed",
            "services",
            "student",
            "technician",
            "unemployed",
            "unknown",
        ],
    )
    marital = st.selectbox(
        "Marital Status", ["married", "single", "divorced", "unknown"]
    )
    education = st.selectbox(
        "Education",
        [
            "basic.4y",
            "basic.6y",
            "basic.9y",
            "high.school",
            "university.degree",
            "professional.course",
            "illiterate",
            "unknown",
        ],
    )
    default = st.selectbox("Has Credit in Default?", ["no", "yes", "unknown"])
    housing = st.selectbox("Has Housing Loan?", ["no", "yes", "unknown"])
    loan = st.selectbox("Has Personal Loan?", ["no", "yes", "unknown"])
    contact = st.selectbox("Contact Type", ["cellular", "telephone"])

with col2:
    month = st.selectbox(
        "Last Contact Month",
        [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ],
    )
    day_of_week = st.selectbox("Day of Week", ["mon", "tue", "wed", "thu", "fri"])
    campaign = st.number_input("# of Contacts (Current Campaign)", min_value=1, value=1)
    pdays = st.number_input("Days Since Last Contact", value=999)
    previous = st.number_input("# of Previous Contacts", min_value=0, value=0)
    poutcome = st.selectbox(
        "Previous Campaign Outcome", ["failure", "nonexistent", "success"]
    )
    emp_var_rate = st.number_input("Employment Variation Rate", value=1.1)
    cons_price_idx = st.number_input("Consumer Price Index", value=93.994)
    cons_conf_idx = st.number_input("Consumer Confidence Index", value=-36.4)
    euribor3m = st.number_input("Euribor 3 Month Rate", value=4.857)
    nr_employed = st.number_input("Number of Employees", value=5191.0)

# --- Manual Encoding ---
job_mapping = {
    "admin.": 0,
    "blue-collar": 1,
    "entrepreneur": 2,
    "housemaid": 3,
    "management": 4,
    "retired": 5,
    "self-employed": 6,
    "services": 7,
    "student": 8,
    "technician": 9,
    "unemployed": 10,
    "unknown": 11,
}
marital_mapping = {"married": 0, "single": 1, "divorced": 2, "unknown": 3}
education_mapping = {
    "basic.4y": 0,
    "basic.6y": 1,
    "basic.9y": 2,
    "high.school": 3,
    "university.degree": 4,
    "professional.course": 5,
    "illiterate": 6,
    "unknown": 7,
}
default_mapping = housing_mapping = loan_mapping = {"no": 0, "yes": 1, "unknown": 2}
contact_mapping = {"cellular": 0, "telephone": 1}
month_mapping = {
    "jan": 0,
    "feb": 1,
    "mar": 2,
    "apr": 3,
    "may": 4,
    "jun": 5,
    "jul": 6,
    "aug": 7,
    "sep": 8,
    "oct": 9,
    "nov": 10,
    "dec": 11,
}
day_mapping = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4}
poutcome_mapping = {"failure": 0, "nonexistent": 1, "success": 2}

input_data = [
    age,
    job_mapping[job],
    marital_mapping[marital],
    education_mapping[education],
    default_mapping[default],
    housing_mapping[housing],
    loan_mapping[loan],
    contact_mapping[contact],
    month_mapping[month],
    day_mapping[day_of_week],
    campaign,
    pdays,
    previous,
    poutcome_mapping[poutcome],
    emp_var_rate,
    cons_price_idx,
    cons_conf_idx,
    euribor3m,
    nr_employed,
]

# --- Predict & Display Result ---
if st.button("🔍 Predict"):
    prediction = model.predict([input_data])[0]
    proba = model.predict_proba([input_data])[0][1]  # Probability of class 1

    st.markdown("### 🌐 Prediction Result")
    if prediction == 1:
        st.success(f"🎉 Likely to SUBSCRIBE (Confidence: {proba:.2%})")
    else:
        st.error(f"❌ Not likely to subscribe (Confidence: {1-proba:.2%})")

    st.markdown("---")
    st.markdown("### 📊 Model Insights")
    st.write(
        "These insights help you understand how well the model performs in real situations:"
    )
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Overall Accuracy", "87.5%")
        st.markdown(
            "<div class='metric-label'>📈 Out of 100 clients, ~88 will be predicted correctly.</div>",
            unsafe_allow_html=True,
        )
    with col4:
        st.metric("Subscription Recall", "48%")
        st.markdown(
            "<div class='metric-label'>🔄 48 out of 100 actual subscribers are identified.</div>",
            unsafe_allow_html=True,
        )
    with col5:
        st.metric("ML Algorithm", "XGBoost")
        st.markdown(
            "<div class='metric-label'>🚀 Fast, efficient and widely used in industry.</div>",
            unsafe_allow_html=True,
        )
