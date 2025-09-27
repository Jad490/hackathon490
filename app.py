import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


#Page Config

st.set_page_config(page_title="❤️ Heart Disease Prediction App ❤️", layout="wide")

st.markdown(
    """
    <style>
    .big-title {
        font-size:40px !important;
        font-weight: bold;
        color: #d6336c;
        text-align: center;
    }
    .sub-title {
        font-size:20px !important;
        text-align: center;
        color: #495057;
    }
    .stButton>button {
        background-color: #d6336c;
        color: white;
        border-radius: 10px;
        font-size: 18px;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #a61e4d;
        color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Loading the Dataset

df = pd.read_csv("heart.csv")

# Encoding
encoder = LabelEncoder()
for col in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
    if col in df.columns:
        df[col] = encoder.fit_transform(df[col])


X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training
model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))


# UI 

st.markdown("<p class='big-title'>❤️ Heart Disease Prediction App</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Enter patient details below to predict likelihood of heart disease.</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 Insights", "📑 Data"])


#Prediction

with tab1:
    col1, col2 = st.columns([1,2])

    with col1:
        st.subheader("🧍 Enter Patient Information")

        age = st.number_input("Age", min_value=1, max_value=120, value=40)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox("Chest Pain Type", [
            "Typical Angina",
            "Atypical Angina",
            "Non-Anginal",
            "Asymptomatic"
        ])
        resting_bp = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
        cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Normal", "High"])
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "LV Hypertrophy"])
        max_hr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    with col2:
        st.subheader("📊 Prediction Result")

        # Map inputs for  dataset encoding
        input_data = pd.DataFrame({
            "Age": [age],
            "Sex": [1 if sex == "Male" else 0],
            "ChestPainType": [ ["Typical Angina","Atypical Angina","Non-Anginal","Asymptomatic"].index(chest_pain) ],
            "RestingBP": [resting_bp],
            "Cholesterol": [cholesterol],
            "FastingBS": [1 if fasting_bs == "High" else 0],
            "RestingECG": [ ["Normal","ST-T Abnormality","LV Hypertrophy"].index(resting_ecg) ],
            "MaxHR": [max_hr],
            "ExerciseAngina": [1 if exercise_angina == "Yes" else 0],
            "Oldpeak": [oldpeak],
            "ST_Slope": [ ["Up","Flat","Down"].index(st_slope) ]
        })

        st.write("### 📝 Patient Data Entered")
        st.dataframe(input_data)

        if st.button("🔍 Predict"):
            prediction = model.predict(input_data)[0]
            confidence = np.max(model.predict_proba(input_data))
            risk_prob = model.predict_proba(input_data)[0][1]  #disease likelyhood

            if prediction == 1:
                st.error("⚠️ Heart Disease Detected")
            else:
                st.success("✅ No Heart Disease Detected")

            # bar 
            st.write(f"**Estimated Risk of Heart Disease: {risk_prob*100:.1f}%**")
            st.progress(int(risk_prob*100))

            st.caption(f"Model confidence: {confidence:.2f}")

        st.info(f"🔮 Model Accuracy: {accuracy:.2f}")


# Graphical Insights 

with tab2:
    st.subheader("📊 Dataset Insights")

    col_a, col_b = st.columns(2)

    with col_a:
        st.write("### Age Distribution by Heart Disease")
        fig, ax = plt.subplots()
        sns.histplot(data=df, x="Age", hue="HeartDisease", bins=20, kde=True, ax=ax)
        st.pyplot(fig)

    with col_b:
        st.write("### Cholesterol Levels by Heart Disease")
        fig, ax = plt.subplots()
        sns.histplot(data=df, x="Cholesterol", hue="HeartDisease", bins=20, kde=True, ax=ax)
        st.pyplot(fig)

    col_c, col_d = st.columns(2)

    with col_c:
        st.write("### Max Heart Rate by Heart Disease")
        fig, ax = plt.subplots()
        sns.histplot(data=df, x="MaxHR", hue="HeartDisease", bins=20, kde=True, ax=ax)
        st.pyplot(fig)

    with col_d:
        st.write("### Resting BP by Heart Disease")
        fig, ax = plt.subplots()
        sns.histplot(data=df, x="RestingBP", hue="HeartDisease", bins=20, kde=True, ax=ax)
        st.pyplot(fig)


#Data

with tab3:
    st.subheader("📑 Raw Dataset Preview")
    st.dataframe(df.head(50))
