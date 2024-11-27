import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Title of the app
st.title("Streamlit Dummy Data Plotting Example")

# Sidebar for user input
st.sidebar.header("Generate Dummy Data")
n_points = st.sidebar.slider("Number of Data Points", 10, 1000, 100)  # Slider to select data points
x_min = st.sidebar.number_input("X-axis Min", value=0.0)
x_max = st.sidebar.number_input("X-axis Max", value=10.0)

# Generate dummy data
x = np.linspace(x_min, x_max, n_points)
y = np.sin(x) + np.random.normal(scale=0.5, size=n_points)  # Add some noise to a sine wave

# Plot the data
fig, ax = plt.subplots()
ax.plot(x, y, label="Dummy Data")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("Dummy Data Plot")
ax.legend()

# Display the plot
st.pyplot(fig)
