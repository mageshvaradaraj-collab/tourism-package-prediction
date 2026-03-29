 

import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="MageshV/tourism-package-prediction", filename="best_tourism_project_model_v1.joblib")

# Load trained model
model = joblib.load(model_path)  # update path if needed

st.title("Tourism Package Prediction")
st.write("Predict whether a customer will purchase a tourism package.")

# -------- USER INPUT -------- #

age = st.number_input("Age", 18, 61, 18)
city_tier = st.selectbox("City Tier", [1, 2, 3])

typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female", "Fe  male"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "UnMarried"])
Designation = st.selectbox("Designation", ["AVP","Executive", "Manager", "Senior Manager", "VP"])

num_persons = st.number_input("Number of Persons Visiting", 1, 5, 1)
preferred_star = st.selectbox("Preferred Hotel Rating", [ 3, 4, 5])
num_trips = st.number_input("Number of Trips per Year", 1, 20, 1)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", 1, 5, 1)
DurationOfPitch = st.number_input("Duration Of Pitch", 5, 127, 5)
NumberOfFollowups = st.number_input("Number Of Followups", 1, 6, 1)
passport = st.selectbox("Has Passport", [0, 1])
own_car = st.selectbox("Own Car", [0, 1])

num_children = st.number_input("Number of Children Visiting", 0, 3, 0)
monthly_income = st.number_input("Monthly Income", 1000, 1000000, 1000)

# -------- CREATE DATAFRAME -------- #

input_data = pd.DataFrame([{
    'Age': age,
    'CityTier': city_tier,
    'TypeofContact': typeofcontact,
    'Occupation': occupation,
    'Gender': gender,
    'MaritalStatus': marital_status,
    'Designation': Designation,
    'NumberOfPersonVisiting': num_persons,
    'PreferredPropertyStar': preferred_star,
    'NumberOfTrips': num_trips,
    'PitchSatisfactionScore' : PitchSatisfactionScore,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfFollowups': NumberOfFollowups,
    'Passport': passport,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': num_children,
    'MonthlyIncome': monthly_income
}])

# -------- PREDICTION -------- #

if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    st.subheader("Result:")
    if prediction == 1:
        st.success("Customer is likely to PURCHASE the package")
    else:
        st.error("Customer is NOT likely to purchase")
