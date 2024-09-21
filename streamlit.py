import numpy as np
import pickle
import streamlit as st

# Load the trained model
loading_model = pickle.load(open('trained_model.sav', 'rb'))  # Ensure the model is in the same directory

# Function for prediction
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loading_model.predict(input_data_reshaped)
    
    return 'The person is not diabetic' if prediction[0] == 0 else 'The person is diabetic'

# Main function for the app
def main():
    # Title of the app
    st.markdown("<h1 style='text-align: center; color: #3498db;'>Diabetes Prediction Web App</h1>", unsafe_allow_html=True)

    # Getting input data from the user
    st.write("### Please enter the following details:")

    # Organize input fields in two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies', '0')
        Glucose = st.text_input('Glucose Level', '0')
        BloodPressure = st.text_input('Blood Pressure value', '0')

    with col2:
        SkinThickness = st.text_input('Skin Thickness value', '0')
        Insulin = st.text_input('Insulin value', '0')
        BMI = st.text_input('BMI value', '0')

    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value', '0.0')
    Age = st.text_input('Age of Person', '0')

    # Code for Prediction
    diagnosis = ''
    
    # Create the button for Prediction
    if st.button('Diabetes Test Result'):
        try:
            # Convert inputs to appropriate types
            input_data = [
                int(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                int(Age)
            ]
            diagnosis = diabetes_prediction(input_data)
            # Display result with color differentiation
            if 'not diabetic' in diagnosis:
                st.success(f'✅ {diagnosis}')
            else:
                st.error(f'⚠️ {diagnosis}')
        except ValueError:
            st.error("Please enter valid numerical values for all fields.")

    st.markdown("---")
    st.markdown("<p style='text-align: center;'>Stay Healthy! 🩺</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
