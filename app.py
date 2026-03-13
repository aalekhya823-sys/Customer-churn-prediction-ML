import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("churn_model.pkl","rb"))

st.title("Customer Churn Prediction")

tenure = st.number_input("Tenure")
monthlycharges = st.number_input("Monthly Charges")
totalcharges = st.number_input("Total Charges")

if st.button("Predict"):
    
    data = np.array([[tenure,monthlycharges,totalcharges]])
    
    prediction = model.predict(data)
    
    if prediction == 1:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer will stay")