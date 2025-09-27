# hackathon490

#Heart Disease Prediction App

##Business Problem
Cardiovascular diseases are the leading cause of death globally. Early detection and risk prediction can help patients and doctors take preventive measures. This project provides a **machine learning–based web application** that predicts the likelihood of heart disease based on patient medical information.

## Dataset
- **Source**: [Heart Disease UCI Dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)  
- **Attributes include**: Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise Induced Angina, ST Depression (Oldpeak), ST Slope  
- **Target variable**: `HeartDisease` → `1` (Disease), `0` (No Disease)

## Approach & Architecture
- Preprocessing: Encoded categorical features using LabelEncoder  
- Model: Random Forest Classifier (scikit-learn)  
- Training: 80/20 train-test split  
- Evaluation: Accuracy ~85–90%  
- Interface: Streamlit for interactive predictions  

**Workflow**:  
User Input → Data Encoding → Random Forest Model → Prediction → Web UI

##How to Run Locally
 Clone the Repository  

git clone https://github.com/Jad490/hackathon490
cd heart-disease-prediction-app
