 

import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.joblib")  # update path if needed

st.title("Tourism Package Prediction")
st.write("Predict whether a customer will purchase a tourism package.")

# -------- USER INPUT -------- #

age = st.number_input("Age", 18, 61, 18)
city_tier = st.selectbox("City Tier", [1, 2, 3])

typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Other"])
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

num_persons = st.number_input("Number of Persons Visiting", 1, 5, 1)
preferred_star = st.selectbox("Preferred Hotel Rating", [ 3, 4, 5])
num_trips = st.number_input("Number of Trips per Year", 1, 20, 1)

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
    'NumberOfPersonVisiting': num_persons,
    'PreferredPropertyStar': preferred_star,
    'NumberOfTrips': num_trips,
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
