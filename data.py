import streamlit as st
import pandas as pd

def dataframe_from_uploaded_files(uploaded_files):
    dataframes = []
    if uploaded_files:
        try:
            for file in uploaded_files:
                df = pd.read_csv(file)
                dataframes.append(df)
        except Exception as e:
            st.error(f"Error reading CSV files: {e}")
            st.stop()
    return dataframes
