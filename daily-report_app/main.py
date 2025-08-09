"""
main.py — Streamlit Audience Analysis Dashboard

This script launches an interactive Streamlit web app that:
- Loads audience data from MongoDB (PVF and IBT)
- Loads pre-trained models from GridFS (KMeans and LSTM)
- Allows the user to select a date and trigger three analysis sections:
    1. Initial data comparison
    2. K-means clustering
    3. LSTM anomaly detection

Run with:
    streamlit run main.py

Dependencies:
    - Streamlit
    - pandas
    - joblib
    - TensorFlow / Keras
    - pymongo
"""



import streamlit as st
from joblib import load
from tensorflow.keras.models import load_model
from loading import ModelLoader, DataFetcher
from k_initial_analysis import (
    clearning,
    plot_unique_household_comparison,
    compare_households_by_age,
    compare_households_by_area
)
from k_cluster_analysis import (
    process_clustering_report,
    plot_weighted_audience_bar,
    plot_deviation_from_average
)

from lstm_analysis import detect_audience_anomalies,display_anomaly_results

# App Title
st.title("📊 Audience Analysis Dashboard")

# Sidebar - Date Input
date_input = st.date_input("📅 Select date for analysis", value=None)

if date_input:
    date = int(date_input.strftime('%Y%m%d'))

    # Load models and data
    with st.spinner("🔄 Loading models and data..."):
        try:
            # Load models
            loader = ModelLoader()
            loader.load_all()

            kmeans_model = loader.kmeans_model
            kmeans_scaler = loader.kmeans_scaler
            lstm_model = loader.lstm_model
            lstm_scaler = loader.lstm_scaler

            # Load data
            fetcher = DataFetcher()
            today_kmeans_df, base_df, lstm_df = fetcher.get_dataframes(date)

            # Clean data
            base_df = clearning(base_df)
            today_kmeans_df = clearning(today_kmeans_df)

        except Exception as e:
            st.error(f"Failed to load data or models: {e}")
            st.stop()

    # Check for empty dataframes
    if base_df.empty or today_kmeans_df.empty:
        st.warning("⚠️ No data available for the selected date.")
        if base_df.empty:
            st.info("Base data is empty.")
        if today_kmeans_df.empty:
            st.info("Today's KMeans data is empty.")
    else:
        st.success("✅ Data and models loaded successfully.")

        # Layout for side-by-side buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            show_initial = st.button("📊 Show Initial daily data Analysis")
        with col2:
            show_kmeans = st.button("🔍 Show K-means Cluster Analysis")
        with col3:
            show_lstm = st.button("📈 LSTM: Spikes and Dips Analysis")


        # Initial Analysis Section
        if show_initial:
            st.header("🏠 Household Comparison Analysis")

            st.subheader("1. Unique Household Comparison")
            fig1 = plot_unique_household_comparison(base_df, today_kmeans_df)
            st.pyplot(fig1)

            st.subheader("2. Household Comparison by Age")
            fig2 = compare_households_by_age(base_df, today_kmeans_df)
            st.pyplot(fig2)

            st.subheader("3. Household Comparison by Area")
            fig3 = compare_households_by_area(base_df, today_kmeans_df)
            st.pyplot(fig3)

        # K-means Cluster Analysis Section
        elif show_kmeans:
            st.header("🧠 K-means Cluster Analysis")

            cluster_summary, cluster_summary_2, area_top3 = process_clustering_report(
                today_kmeans_df, kmeans_scaler, kmeans_model
            )

            st.subheader("1. Weighted Audience Size per Cluster")
            fig4 = plot_weighted_audience_bar(cluster_summary_2)
            st.pyplot(fig4)

            st.subheader("2. Deviation from Average Viewing Patterns")
            fig5 = plot_deviation_from_average(cluster_summary_2)
            st.pyplot(fig5)

            st.subheader("3. Cluster Summary")
            st.dataframe(cluster_summary)

            st.subheader("4. Top 3 Areas per Cluster")
            st.dataframe(area_top3)

        # LSTM Spike and Dip Analysis Section
        elif show_lstm:
            st.header("📈 LSTM: Spikes and Dips Anomaly Detection")

            df_anomalies, df_pred = detect_audience_anomalies(
                    lstm_df, lstm_model, lstm_scaler, lookback=60
             )
            lstm_visual, df_anomalies_table = display_anomaly_results(df_pred, df_anomalies)

            st.subheader("1. Detected Anomalies")
            st.dataframe(df_anomalies_table)

            st.subheader("2. Audience Prediction with Anomalies Highlighted")
            st.pyplot(lstm_visual)


        else:
            st.info("Click a button above to display an analysis section.")
else:
    st.info("Please select a date to begin the analysis.")

