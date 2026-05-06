import streamlit as st
from src.predict import predict_price

st.title("House Price Prediction")

sqft = st.number_input("Square Footage", min_value=500)
bedrooms = st.number_input("Bedrooms", min_value=1)
bathrooms = st.number_input("Bathrooms", min_value=1)

if st.button("Predict Price"):
    price = predict_price(sqft, bedrooms, bathrooms)
    st.success(f"Estimated Price: ₹ {price:,.2f}")