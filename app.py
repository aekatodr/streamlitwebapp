# import libraries
import streamlit as st
from joblib import load
import numpy as np

# load the model from disk
model = load('iris_model.joblib')

# create streamlit Web app
st.title('Iris Flower Prediction App')

# create streamlit UI

# create input field for user to enter
sepal_length = st.slider('Sepal Length', 4.3, 7.9, 5.4)
sepal_width = st.slider('Sepal Width', 2.0, 4.4, 3.4)
petal_length = st.slider('Petal Length', 1.0, 6.9, 1.6)
petal_width = st.slider('Petal Width', 0.1, 2.5, 0.2)

features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# create button to predict
if st.button('Predict'):
    prediction = model.predict(features)

    st.write(f"The Flower species is:', {prediction[0]}")
   