# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 19:06:51 2026

@author: ADMIN
"""

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier

# ---------------- LOAD DATA ----------------
df = pd.read_csv(r"Crop_recommendation.csv")

# ---------------- CROP MODEL ----------------
X = df[['N','P','K','temperature','humidity','ph','rainfall']]
y = df['label']

model = RandomForestClassifier()
model.fit(X, y)

# ---------------- IRRIGATION MODEL ----------------
df['irrigation'] = df['humidity'].apply(lambda x: 1 if x < 40 else 0)

X_irrigation = df[['humidity','temperature']]
y_irrigation = df['irrigation']

model_irrigation = DecisionTreeClassifier()
model_irrigation.fit(X_irrigation, y_irrigation)

# ---------------- REVENUE MODEL ----------------
df['production'] = np.random.randint(1000, 5000, size=len(df))
df['price'] = np.random.randint(10, 100, size=len(df))
df['revenue'] = df['production'] * df['price']

X_rev = df[['production','price']]
y_rev = df['revenue']

model_revenue = RandomForestRegressor()
model_revenue.fit(X_rev, y_rev)

# ---------------- STREAMLIT UI ----------------
st.title("🌱 Smart Agriculture DSS")

st.write("Enter soil and weather details:")

N = st.number_input("Nitrogen (N)", min_value=0)
P = st.number_input("Phosphorus (P)", min_value=0)
K = st.number_input("Potassium (K)", min_value=0)

temp = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("pH value")
rainfall = st.number_input("Rainfall (mm)")

if st.button("Predict"):

    # Crop Prediction
    crop_input = pd.DataFrame([[N, P, K, temp, humidity, ph, rainfall]],
                              columns=['N','P','K','temperature','humidity','ph','rainfall'])
    crop = model.predict(crop_input)[0]

    # Irrigation Prediction
    irrigation_input = pd.DataFrame([[humidity, temp]],
                                    columns=['humidity','temperature'])
    irrigation = model_irrigation.predict(irrigation_input)[0]

    # Revenue Prediction
    production = 3000
    price = 50

    revenue_input = pd.DataFrame([[production, price]],
                                 columns=['production','price'])
    revenue = model_revenue.predict(revenue_input)[0]

    # Output
    st.subheader("🌾 Results")
    st.write("🌱 Recommended Crop:", crop)
    st.write("💧 Irrigation Needed:", "Yes" if irrigation==1 else "No")
    st.write("💰 Expected Revenue:", revenue)
