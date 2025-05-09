import streamlit as st

st.title("New Your Housing Price Prediction")

st.write('This is a simple web app to predict the price of a house in New York City based on its size in square feet and the number of bedrooms.')

bedrooms = st.number_input("The number of bedrooms", value=0, step=1, format="%d")
bathrooms = st.number_input("Bathrooms: ", value=0, step=1, format="%d")
size = st.number_input("Size (sqft): ", value=0, step=50, format="%d")


st.write("Bedrooms:", bedrooms)
st.write("Bathrooms:", bathrooms)
st.write("Size in sqft:", size)