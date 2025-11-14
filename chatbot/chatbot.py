import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import OrdinalEncoder
import pickle

st.set_page_config(page_title="🧠 Stroke Prediction App", page_icon="💉", layout="centered")

if 'clear_form' not in st.session_state:
    st.session_state.clear_form = False

# -----------------------------
# Load model, dataset, and mean BMI
# -----------------------------
with st.spinner("Loading model and data..."):
    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'xgb_boost_model.pk1')
        loaded_model = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}")
        st.stop()

    try:
        df2 = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'train', 'train_results', 'dataset_dataframe.csv'))
    except FileNotFoundError:
        st.warning("CSV dataset not found. Some features may not work correctly.")
        df2 = None

    try:
        with open(os.path.join(os.path.dirname(__file__), '..', 'train', 'train_results', 'mean_bmi.pkl'), "rb") as f:
            mean_bmi = pickle.load(f)
    except FileNotFoundError:
        st.warning("mean_bmi.pkl not found. Missing BMI values will be treated as 25.0")
        mean_bmi = 25.0

st.success("✅ XGBoost model loaded successfully!")

# -----------------------------
# Page title & sidebar
# -----------------------------
st.markdown(
    """
    <div style='text-align:center'>
        <h1>🧠 Stroke Prediction App</h1>
        <p style='color:gray;'>Predict your stroke risk using AI-powered analysis.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("💡 About")
st.sidebar.info(
    "Predict your stroke risk based on health indicators.\n\n"
    "Adjust inputs and click **Predict Stroke** to see results."
)
st.sidebar.markdown("---")
st.sidebar.write("👨‍⚕️ Model: XGBoost Classifier")
st.sidebar.write("📊 Trained with real patient data by the AI-Stroke-Shield team")
st.sidebar.write("""🩺 Built by:  
- Sidney Mpenyana  
- SG Rakobela  
- VP Machave  
- P Chauke
""")

# -----------------------------
# User input form
# -----------------------------
with st.form(key='stroke_form'):
    st.markdown("### 🧍‍♂️ Demographic Information")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("⚥ Gender", ["Male", "Female"], index=0 if st.session_state.clear_form else 0)
        age = st.number_input("🎂 Age", min_value=0, max_value=100, value=0 if st.session_state.clear_form else 50)
        ever_married = st.selectbox("💍 Ever Married", ["Yes", "No"], index=0 if st.session_state.clear_form else 0)
    with col2:
        residence_type = st.selectbox("🏡 Residence Type", ["Urban", "Rural"], index=0 if st.session_state.clear_form else 0)
        work_type = st.selectbox("💼 Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"], index=0 if st.session_state.clear_form else 0)

    st.markdown("### ❤️ Health Indicators")
    col3, col4 = st.columns(2)
    with col3:
        hypertension = st.selectbox("🩸 Hypertension", ['No', 'Yes'], index=0 if st.session_state.clear_form else 0)
        heart_disease = st.selectbox("💔 Heart Disease", ['No', 'Yes'], index=0 if st.session_state.clear_form else 0)
    with col4:
        avg_glucose_level = st.number_input("🧪 Average Glucose Level", min_value=0.0, max_value=500.0, value=0.0 if st.session_state.clear_form else 120.0)
        bmi = st.number_input("⚖️ BMI", min_value=0.0, max_value=100.0, value=0.0 if st.session_state.clear_form else 25.0)

    smoking_status = st.selectbox("🚬 Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"], index=0 if st.session_state.clear_form else 0)

    submit_button = st.form_submit_button(label='🔍 Predict Stroke')
    clear_button = st.form_submit_button(label='🧹 Clear Form')

# -----------------------------
# Clear form functionality
# -----------------------------
if clear_button:
    st.session_state.clear_form = True
    st.experimental_rerun()  

if submit_button:
    st.session_state.clear_form = False

    # Convert categorical to numerical
    hypertension = 1 if hypertension == "Yes" else 0
    heart_disease = 1 if heart_disease == "Yes" else 0
    ever_married = 1 if ever_married == "Yes" else 0
    residence_type = 1 if residence_type == "Urban" else 0

    new_user_data = {
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status]
    }

    new_user_df = pd.DataFrame(new_user_data)
    new_user_df['bmi'] = pd.to_numeric(new_user_df['bmi'], errors='coerce').fillna(mean_bmi)

    # Ordinal encode categorical features
    if df2 is not None:
        categorical_cols = ['gender', 'work_type', 'smoking_status']
        combined_df_for_fitting = pd.concat(
            [df2[categorical_cols].astype(str), new_user_df[categorical_cols].astype(str)],
            ignore_index=True
        )
        oe = OrdinalEncoder()
        oe.fit(combined_df_for_fitting)
        new_user_df[categorical_cols] = oe.transform(new_user_df[categorical_cols])

    # Predict
    predicted_class = loaded_model.predict(new_user_df)[0]
    predicted_proba = float(loaded_model.predict_proba(new_user_df)[:, 1][0])

    # Display results
    st.markdown("---")
    st.markdown("## 🧾 Prediction Results")

    if predicted_class == 1:
        st.error(f"⚠️ High risk of Stroke!\n\n**Probability:** {predicted_proba:.2%}")
        st.markdown("### 🩺 Stroke Prevention Tips")
        st.info("""- 🧂 Reduce salt intake  
- 🥗 Eat a balanced diet  
- 🚶 Exercise regularly  
- 🚭 Quit smoking  
- 🍷 Limit alcohol  
- 💊 Manage chronic conditions  
- 🧘 Stress reduction  
- 🩺 Regular check-ups""")
        st.markdown(
            "<p style='color:red; font-size:0.85rem;'>Source: WHO, CDC, American Heart Association.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.success(f"✅ Low risk of Stroke.\n\n**Probability:** {predicted_proba:.2%}")
        st.markdown("### ✅ Keep Up the Good Habits!")
        st.info("""- Maintain a healthy diet and lifestyle  
- Avoid smoking/alcohol  
- Monitor BP and glucose  
- Regular check-ups  
- Manage stress""")
        st.markdown(
            "<p style='color:green; font-size:0.85rem;'>Source: WHO, CDC, American Heart Association.</p>",
            unsafe_allow_html=True,
        )

    st.progress(predicted_proba)
