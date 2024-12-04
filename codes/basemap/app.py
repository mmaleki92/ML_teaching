import streamlit as st
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
from copy import deepcopy
from cartopy.util import add_cyclic_point

# Function to plot the dataset
def plot_dataset(dataset: xr.Dataset, lon_min, lon_max, lat_min, lat_max):
    """
    Plot temperature anomaly data on a Mercator projection map.

    Parameters:
        dataset (xr.Dataset): The dataset containing 't2m' data and 'valid_time'.
        lon_min, lon_max, lat_min, lat_max (float): Longitude and latitude limits for the map.
    """
    # Define the map projection (Mercator) for visualization
    projection = ccrs.Mercator()

    # CRS of the dataset (PlateCarree assumes data is in lat/lon)
    data_crs = ccrs.PlateCarree()

    # Create a figure and map axes using the Mercator projection
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150, subplot_kw={'projection': projection})

    # Add gridlines (latitude and longitude lines) to the map
    gridlines = ax.gridlines(
        crs=data_crs, draw_labels=True,
        linewidth=0.6, color='gray', alpha=0.5, linestyle='-.'
    )
    gridlines.xlabel_style = {"size": 7}
    gridlines.ylabel_style = {"size": 7}

    # Add geographical features: coastlines and country borders
    ax.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.5)
    ax.add_feature(cf.BORDERS.with_scale("50m"), lw=0.3)

    # Define the geographic extent of the map
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=data_crs)

    # Plot the temperature anomaly ('t2m') data on the map
    cbar_kwargs = {
        'orientation': 'horizontal', 'shrink': 0.6, "pad": 0.05,
        'aspect': 40, 'label': '2 Metre Temperature Anomaly [K]'
    }
    dataset["t2m"].plot.contourf(
        ax=ax, transform=data_crs, cbar_kwargs=cbar_kwargs, levels=21
    )

    # Add a title with the time information from the dataset
    plt.title(f"Temperature Anomaly for {dataset.valid_time.dt.strftime('%B %Y').values}")

    # Display the plot
    plt.show()


# Streamlit App
st.title("Temperature Anomaly Visualization")

# Upload the data file
uploaded_file = "data/1month_anomaly_Global_ea_2t_201907_1991-2020_v02.grib"#st.file_uploader("Upload the GRIB data file", type=["grib"])

# Input for latitude and longitude ranges
lon_min = st.slider("Select Longitude Range - Minimum", -180.0, 180.0, 44.0)
lon_max = st.slider("Select Longitude Range - Maximum", -180.0, 180.0, 63.0)
lat_min = st.slider("Select Latitude Range - Minimum", -90.0, 90.0, 25.0)
lat_max = st.slider("Select Latitude Range - Maximum", -90.0, 90.0, 40.0)

if uploaded_file is not None:
    # Open the dataset from the uploaded file
    original_data = xr.open_dataset(uploaded_file, engine="cfgrib")

    # Create a copy and crop it to the selected lat-lon range
    cropped_dataset = deepcopy(original_data)
    cropped_dataset = cropped_dataset.sel(
        latitude=slice(lat_max, lat_min), 
        longitude=slice(lon_min, lon_max)
    )

    # Display the plot
    st.pyplot(plot_dataset(cropped_dataset, lon_min, lon_max, lat_min, lat_max))
