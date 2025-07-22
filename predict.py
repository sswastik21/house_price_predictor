import joblib
import pandas as pd

# Load model
model = joblib.load('house_price_model.pkl')

print("🏠 Welcome to House Price Predictor")
print("Enter the following details:")

# Take input from user
MedInc = float(input("Median Income: "))
HouseAge = float(input("House Age: "))
AveRooms = float(input("Average Rooms: "))
AveBedrms = float(input("Average Bedrooms: "))
Population = float(input("Population: "))
AveOccup = float(input("Average Occupancy: "))
Latitude = float(input("Latitude: "))
Longitude = float(input("Longitude: "))

# Prepare input as DataFrame
input_data = pd.DataFrame([{
    'MedInc': MedInc,
    'HouseAge': HouseAge,
    'AveRooms': AveRooms,
    'AveBedrms': AveBedrms,
    'Population': Population,
    'AveOccup': AveOccup,
    'Latitude': Latitude,
    'Longitude': Longitude
}])

# Predict
price = model.predict(input_data)
print(f"✅ Estimated House Price: ${price[0] * 100000:.2f}")
