
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt



def detect_audience_anomalies(df, model, scaler, lookback=60):
    """
    Detect audience anomalies (spikes and dips) using an LSTM model.

    This function:
    1. Processes DATE and TIME columns into a full datetime.
    2. Scales the audience values using the provided scaler.
    3. Creates sequential lookback windows for LSTM prediction.
    4. Uses the provided model to predict audience values.
    5. Calculates residuals (actual - predicted).
    6. Flags anomalies using an Interquartile Range (IQR) threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        Input time series data with columns:
        - 'DATE': date of observation (string or datetime)
        - 'TIME': time in HHMM integer format
        - 'AUDIENCE': numeric audience values
        Optional columns 'Panel' and 'DATA_TYPE' will be dropped if present.
    model : keras.Model or similar
        Pre-trained LSTM model used for prediction.
    scaler : sklearn.preprocessing object
        Fitted scaler for transforming and inverse transforming audience values.
    lookback : int, optional
        Number of previous timesteps to use for each LSTM input sequence (default=60).

    Returns
    -------
    tuple of pandas.DataFrame
        df_anomalies : subset of rows flagged as 'dip' or 'spike'.
        df_pred : full dataset with columns:
            - 'PREDICTED_AUDIENCE': model predictions
            - 'RESIDUAL': difference between actual and predicted
            - 'ANOMALY': 'dip', 'spike', or 'normal'

    """

    # Step 1: Convert TIME and DATE to full datetime
    def convert_time_columns(df):
        df['HOUR'] = df['TIME'] // 100
        df['MINUTE'] = df['TIME'] % 100
        df['DATE'] = pd.to_datetime(df['DATE'])  # ensure datetime
        df['ADJUSTED_DATE'] = df.apply(
            lambda row: row['DATE'] + pd.Timedelta(days=1) if row['HOUR'] >= 24 else row['DATE'],
            axis=1
        )
        df['DATE'] = pd.to_datetime(df['ADJUSTED_DATE'].dt.date.astype(str)) + \
                     pd.to_timedelta(df['HOUR'] % 24, unit='h') + \
                     pd.to_timedelta(df['MINUTE'], unit='m')
        df.drop(columns=['HOUR', 'MINUTE', 'ADJUSTED_DATE'], inplace=True)
        return df

    # Step 2: Initial processing
    df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True, errors='coerce')
    df = df.drop(columns=['Panel', 'DATA_TYPE'], errors='ignore')
    df = convert_time_columns(df)
     #df['AUDIENCE'] = df['AUDIENCE'].astype(str).str.replace(',', '', regex=False).astype(float)
    df = df.sort_values('DATE')


    # Step 4: Scale and prepare sequences
    audience_scaled = scaler.transform(df[['AUDIENCE']])
    X = np.array([audience_scaled[i - lookback:i] for i in range(lookback, len(audience_scaled))])

    # Step 5: Predict and align
    pred_scaled = model.predict(X)
    pred = scaler.inverse_transform(pred_scaled)
    df_pred = df.iloc[lookback:].copy()
    df_pred['PREDICTED_AUDIENCE'] = pred
    df_pred['RESIDUAL'] = df_pred['AUDIENCE'] - df_pred['PREDICTED_AUDIENCE']

    # Step 6: Anomaly detection using IQR
    Q1 = df_pred['RESIDUAL'].quantile(0.25)
    Q3 = df_pred['RESIDUAL'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 15 * IQR
    upper_bound = Q3 + 15 * IQR
    df_pred['ANOMALY'] = df_pred['RESIDUAL'].apply(
        lambda x: 'dip' if x < lower_bound else ('spike' if x > upper_bound else 'normal')
    )

    # Step 7: Return anomalies only
    df_anomalies = df_pred[df_pred['ANOMALY'].isin(['dip', 'spike'])]

    return df_anomalies, df_pred



def display_anomaly_results(df_pred, df_anomalies):
    """
    Plot audience data with predicted values and highlight anomalies.

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Full dataset with 'DATE', 'AUDIENCE', 'PREDICTED_AUDIENCE', and 'ANOMALY' columns.
    df_anomalies : pandas.DataFrame
        Subset of anomalies containing 'TIME', 'AUDIENCE', 'PREDICTED_AUDIENCE', and 'ANOMALY'.

    Returns
    -------
    tuple
        fig : matplotlib.figure.Figure
            The generated anomaly plot.
        df_anomalies_table : pandas.DataFrame
            Formatted anomaly table with rounded values.
    """

    # Step 1: Create plot
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.plot(df_pred['DATE'], df_pred['AUDIENCE'], label='Actual', alpha=0.6)
    ax.plot(df_pred['DATE'], df_pred['PREDICTED_AUDIENCE'], label='Predicted', alpha=0.8)

    ax.scatter(df_pred[df_pred['ANOMALY'] == 'dip']['DATE'],
               df_pred[df_pred['ANOMALY'] == 'dip']['AUDIENCE'],
               color='blue', label='Dip')

    ax.scatter(df_pred[df_pred['ANOMALY'] == 'spike']['DATE'],
               df_pred[df_pred['ANOMALY'] == 'spike']['AUDIENCE'],
               color='red', label='Spike')

    ax.set_title("Audience Anomaly Detection")
    ax.set_xlabel("Date")
    ax.set_ylabel("Audience")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    # Step 2: Format df_anomalies
    df_anomalies_table = df_anomalies[['TIME', 'AUDIENCE', 'PREDICTED_AUDIENCE', 'ANOMALY']].copy()
    df_anomalies_table['PREDICTED_AUDIENCE'] = df_anomalies_table['PREDICTED_AUDIENCE'].round(2)
    df_anomalies_table['AUDIENCE'] = df_anomalies_table['AUDIENCE'].round(2)
    df_anomalies_table.reset_index(drop=True, inplace=True)

    return fig, df_anomalies_table
