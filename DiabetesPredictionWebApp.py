# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 01:30:46 2024

@author: Success
"""

import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loading_model = pickle.load(open('C:/ML/Diabetes_prediction/trained_model.sav', 'rb'))

# Function for prediction
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loading_model.predict(input_data_reshaped)
    
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

# Main function for the app
def main():
    # Giving the title
    st.markdown("<h1 style='text-align: center; color: #3498db;'>Diabetes Prediction Web App</h1>", unsafe_allow_html=True)

    # Getting input data from the user
    st.write("### Please enter the following details:")

    # Organize input fields in two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        Glucose = st.text_input('Glucose Level')
        BloodPressure = st.text_input('Blood Pressure value')

    with col2:
        SkinThickness = st.text_input('Skin Thickness value')
        Insulin = st.text_input('Insulin value')
        BMI = st.text_input('BMI value')

    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of Person')

    # Code for Prediction
    diagnosis = ''
    
    # Creating the button for Prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        # Display result with color differentiation for clarity
        if 'not diabetic' in diagnosis:
            st.success(f'✅ {diagnosis}')
        else:
            st.error(f'⚠️ {diagnosis}')
    
    st.markdown("---")
    st.markdown("<p style='text-align: center;'>Stay Healthy! 🩺</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
