# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle


# Loading the saved model

loading_model = pickle.load(open('C:/ML/Diabetes_prediction/trained_model.sav','rb'))



input_data = (0,137,40,35,168,43.1,2.288,33) 

# changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) 

prediction =loading_model.predict(input_data_reshaped)

print(prediction)

if(prediction[0]==0):
  print('Person is not diabetic')
else:
  print('The person is diabetic')
