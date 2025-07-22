import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('house_price_model.pkl')

st.title("🏠 House Price Predictor (California)")
st.markdown("Enter house features below:")

# Collect user input
MedInc = st.number_input("Median Income", value=5.0)
HouseAge = st.number_input("House Age", value=20.0)
AveRooms = st.number_input("Average Rooms", value=5.0)
AveBedrms = st.number_input("Average Bedrooms", value=1.0)
Population = st.number_input("Population", value=1000.0)
AveOccup = st.number_input("Average Occupancy", value=3.0)
Latitude = st.number_input("Latitude", value=34.0)
Longitude = st.number_input("Longitude", value=-118.0)

# Predict when button is clicked
if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        'MedInc': MedInc,
        'HouseAge': HouseAge,
        'AveRooms': AveRooms,
        'AveBedrms': AveBedrms,
        'Population': Population,
        'AveOccup': AveOccup,
        'Latitude': Latitude,
        'Longitude': Longitude
    }])

    prediction = model.predict(input_df)
    price = prediction[0] * 100000  # scale to dollars

    st.success(f"💰 Estimated House Price: ${price:,.2f}")
