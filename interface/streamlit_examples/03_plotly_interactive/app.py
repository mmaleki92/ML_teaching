import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# App Title
st.title("Interactive 3D Plot with Plotly")
st.write("Explore 3D visualizations generated from customizable dummy data.")

# Sidebar for customization
st.sidebar.header("3D Data Settings")
n_points = st.sidebar.slider("Number of Data Points", min_value=100, max_value=5000, value=1000, step=100)
distribution = st.sidebar.selectbox("Select Data Distribution", ["Normal", "Uniform", "Helix", "Sphere"])
random_seed = st.sidebar.number_input("Random Seed", value=42, step=1)

# Set random seed for reproducibility
np.random.seed(random_seed)

# Generate dummy data
if distribution == "Normal":
    x = np.random.normal(size=n_points)
    y = np.random.normal(size=n_points)
    z = np.random.normal(size=n_points)
elif distribution == "Uniform":
    x = np.random.uniform(-1, 1, size=n_points)
    y = np.random.uniform(-1, 1, size=n_points)
    z = np.random.uniform(-1, 1, size=n_points)
elif distribution == "Helix":
    theta = np.linspace(0, 4 * np.pi, n_points)
    z = np.linspace(-2, 2, n_points)
    x = np.sin(theta)
    y = np.cos(theta)
elif distribution == "Sphere":
    phi = np.random.uniform(0, np.pi, size=n_points)
    theta = np.random.uniform(0, 2 * np.pi, size=n_points)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

# Create DataFrame for easier handling
df = pd.DataFrame({"X": x, "Y": y, "Z": z})

# Sidebar for visualization settings
st.sidebar.header("3D Plot Settings")
plot_type = st.sidebar.radio("Plot Type", ["Scatter", "Surface (Requires Grid Data)"])
color_by = st.sidebar.selectbox("Color By", ["None", "X", "Y", "Z"])

# Create interactive 3D plot
st.subheader("3D Plot")
if plot_type == "Scatter":
    color = None if color_by == "None" else df[color_by]
    fig = px.scatter_3d(df, x="X", y="Y", z="Z", color=color, title="3D Scatter Plot")
    fig.update_traces(marker=dict(size=3))  # Adjust marker size
else:
    # Surface requires gridded data, create a grid from X and Y
    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), int(np.sqrt(n_points))),
        np.linspace(y.min(), y.max(), int(np.sqrt(n_points))),
    )
    grid_z = np.sin(np.sqrt(grid_x**2 + grid_y**2))  # Example function for surface
    fig = go.Figure(data=[go.Surface(z=grid_z, x=grid_x, y=grid_y)])
    fig.update_layout(title="3D Surface Plot", scene=dict(zaxis=dict(range=[-2, 2])))

# Show plot
st.plotly_chart(fig, use_container_width=True)

# Display data sample
st.subheader("Generated Data Sample")
st.write(df.head())

