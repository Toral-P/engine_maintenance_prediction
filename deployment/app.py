import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download

import joblib
# Download and load the trained model
model_path = hf_hub_download(repo_id="toriaiml/Vehicles-Predictive-Maintenance", filename="predictive_maintenance_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Predictive Manintenance - Vehicles")
st.write("""
This application predicts whether the engine requires maintenance or is operating normally.
Please enter the app details below to get the prediction:
""")

# User input

EngineRPM = st.number_input("Engine RPM", min_value=10, max_value=3000)
LubOilPressure = st.number_input("Lubricant Oil pressure", min_value=0, max_value=10)
FuelPressure = st.number_input("Fuel Pressure", min_value=0, max_value=20)
CoolantPressure = st.number_input("Coolant Pressure", min_value=0, max_value=9)
LubOilTemp = st.number_input("Lubricant Oil Temperature", min_value=70, max_value=90)
CoolantTemp = st.number_input("Coolant Temperature", min_value=50, max_value=100)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Engine rpm': EngineRPM,
    'Lub oil pressure': LubOilPressure,
    'Fuel pressure': FuelPressure,
    'Coolant pressure': CoolantPressure,
    'lub oil temp': LubOilTemp,
    'Coolant temp': CoolantTemp
}])

# Predict button
if st.button("Predict Engine Condition"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Predicted Engine Condition: {prediction}")
