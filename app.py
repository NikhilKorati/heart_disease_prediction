import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the model
# pipeline_path = os.path.join('pipeline.joblib')
### for now commented the pipeline part as model is being used just for prediction 
### so pipeline is an overkill, use it when you have multiple steps before and after prediction

model_path = os.path.join('hdp_model.pkl')

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        st.success(f"Successfully loaded model from {model_path}")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
else:
    st.error(f"Model file not found at {model_path}. Please check the file path.")

def predict(features):
    if model is None:
        st.error("Model not loaded. Cannot make prediction.")
        return None, None
    try:
        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features)
        return prediction, probability
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None

def main():
    st.title('Heart Disease Prediction App')

    st.write("""
    This app predicts the likelihood of heart disease based on several health indicators.
    Please enter your health information below:
    """)

    # Create input fields for features
    age = st.number_input('Age', 1, 120, 25)
    cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
    trestbps = st.number_input('Resting Blood Pressure (mm Hg)', 90, 200, 120)
    chol = st.number_input('Serum Cholesterol (mg/dl)', 100, 600, 200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
    restecg = st.selectbox('Resting ECG Results', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
    thalach = st.number_input('Maximum Heart Rate Achieved', 60, 220, 150)
    exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
    oldpeak = st.number_input('ST Depression Induced by Exercise', 0.0, 6.0, 0.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', [1, 2, 3])
    ca = st.number_input('Number of Major Vessels Colored by Fluoroscopy', 0, 4, 0)
    thal = st.selectbox('Thalassemia', [1, 2, 3])

    # Convert categorical features to numerical
    cp_dict = {'Typical Angina': 1, 'Atypical Angina': 2, 'Non-anginal Pain': 3, 'Asymptomatic': 4}
    fbs_dict = {'No': 0, 'Yes': 1}
    restecg_dict = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
    exang_dict = {'No': 0, 'Yes': 1}

    input_data = pd.DataFrame({
        'cp': [cp_dict[cp]],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs_dict[fbs]],
        'restecg': [restecg_dict[restecg]],
        'thalach': [thalach],
        'exang': [exang_dict[exang]],
    })

    # Make prediction when button is clicked
    if st.button('Predict'):
        prediction, probability = predict(input_data)
        
        if prediction is not None:
            st.subheader('Prediction')
            if prediction[0] == 1:
                st.write('High likelihood of heart disease')
            else:
                st.write('Low likelihood of heart disease')
            
            st.subheader('Prediction Probability')
            st.write(f'Probability of heart disease: {probability[0][1]:.2f}')

            # Visualize feature importance
            st.subheader('Feature Importance')
            # feature_importance = model.named_steps['classifier'].feature_importances_
            # feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            ### Commented lines applicable to tree based models, here logistic regeression is used so made some changes
            feature_importance = model.coef_

            fig, ax = plt.subplots()
            sns.barplot(x=feature_importance[0], y=input_data.columns, ax=ax)
            plt.title('Feature Importance')
            st.pyplot(fig)

        else:
            st.error("Prediction could not be made.")

if __name__ == '__main__':
    main()
