import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the Streamlit app
st.title("Interactive Data Visualization with Streamlit")
st.write("An intermediate example with customizable dummy data and multiple visualization options.")

# Sidebar for user input
st.sidebar.header("Data Settings")
n_points = st.sidebar.slider("Number of Data Points", min_value=50, max_value=2000, value=500, step=50)
distribution = st.sidebar.selectbox("Select Data Distribution", ["Normal", "Uniform", "Exponential"])
seed = st.sidebar.number_input("Random Seed (Optional)", value=42, step=1)

# Generate dummy data based on selected distribution
np.random.seed(seed)
if distribution == "Normal":
    data = np.random.normal(loc=0, scale=1, size=n_points)
elif distribution == "Uniform":
    data = np.random.uniform(low=-1, high=1, size=n_points)
elif distribution == "Exponential":
    data = np.random.exponential(scale=1.0, size=n_points)

# Create a DataFrame for better handling
df = pd.DataFrame({"Index": np.arange(1, n_points + 1), "Value": data})

# Visualization options
st.sidebar.header("Plot Settings")
plot_type = st.sidebar.radio("Choose a Plot Type", ["Line Plot", "Scatter Plot", "Histogram", "Box Plot"])
show_grid = st.sidebar.checkbox("Show Grid", value=True)
color = st.sidebar.color_picker("Pick a Plot Color", value="#1f77b4")

# Plot the data
st.subheader("Visualization")
fig, ax = plt.subplots()

if plot_type == "Line Plot":
    ax.plot(df["Index"], df["Value"], color=color, label="Line Plot")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
elif plot_type == "Scatter Plot":
    ax.scatter(df["Index"], df["Value"], color=color, label="Scatter Plot")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
elif plot_type == "Histogram":
    sns.histplot(df["Value"], kde=True, color=color, ax=ax)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
elif plot_type == "Box Plot":
    sns.boxplot(data=df, y="Value", color=color, ax=ax)
    ax.set_ylabel("Value")

ax.set_title(f"{plot_type} of Generated Data")
if show_grid:
    ax.grid(True)

# Display the plot
st.pyplot(fig)

# Display data table
st.subheader("Generated Data")
st.write("Here is a preview of the generated dataset:")
st.dataframe(df.head(10))

