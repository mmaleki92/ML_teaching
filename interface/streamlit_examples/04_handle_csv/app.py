import streamlit as st
import pandas as pd

# App Title
st.title("CSV Reader and Uploader")
st.write("This app demonstrates how to read a CSV file and upload one interactively.")

# Example for reading a CSV from a file
st.subheader("Read CSV from File")
st.write("This example uses a sample CSV file included in the app.")

# Load a sample CSV
sample_csv = "https://people.sc.fsu.edu/~jburkardt/data/csv/airtravel.csv"
df_sample = pd.read_csv(sample_csv)

st.write("Here is the sample CSV content:")
st.dataframe(df_sample)

# Uploading CSV file via Streamlit
st.subheader("Upload Your CSV")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    # Read the uploaded file
    df_uploaded = pd.read_csv(uploaded_file)

    # Display the first few rows
    st.write("Here are the first few rows of your uploaded CSV:")
    st.dataframe(df_uploaded)

    # Optional: Show summary statistics
    st.write("Summary Statistics:")
    st.write(df_uploaded.describe())

    # Optionally allow the user to download the uploaded data as CSV
    st.subheader("Download Your Uploaded Data")
    csv = df_uploaded.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="uploaded_data.csv",
        mime="text/csv",
    )
else:
    st.write("Upload a CSV file to see its contents here!")
