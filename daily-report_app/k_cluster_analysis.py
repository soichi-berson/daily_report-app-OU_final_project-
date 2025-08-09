


import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

def process_clustering_report(df, SCALER_PATH, KMEANS_PATH):
    """
    Generate clustering-based viewing behaviour reports.

    This function:
    1. Aggregates and pivots viewing session data by household and genre.
    2. Calculates genre ratios, peak viewing time, and total recording usage.
    3. Creates feature groups (Leisure, Factual, Sport, VOD) for clustering.
    4. Applies a pre-fitted scaler and KMeans model to assign clusters.
    5. Summarises each cluster with weighted averages and interpretations.
    6. Produces weighted audience share reports and top areas per cluster.

    Parameters
    ----------
    df : pandas.DataFrame
        Viewing session dataset. Must include columns:
        'Household number', 'Genre', 'Duration of session',
        'Recording', 'viewing time', 'Area', 'Age', 'Processing Weight'.
    SCALER_PATH : object 
        Pre-fitted scaler instance used to normalise feature values.
    KMEANS_PATH : object 
        Pre-fitted KMeans instance used to assign clusters.

    Returns
    -------
    tuple of pandas.DataFrame
        cluster_summary : Cluster-level metrics and interpretations.
        summary : Weighted audience size and percentage share by cluster.
        pivot : Top 3 areas for each cluster with weighted audience figures.

    """


    # --- STEP 1: Pivot and Aggregate ---
    pivot_df = df.pivot_table(index='Household number', columns='Genre', values='Duration of session', aggfunc='sum', fill_value=0).reset_index()
    total_duration_df = df.groupby('Household number')['Duration of session'].sum().reset_index(name='Total Duration')
    grouped = df.groupby(['Household number', 'viewing time'])['Duration of session'].sum().reset_index()
    peak_viewing = grouped.loc[grouped.groupby('Household number')['Duration of session'].idxmax()]
    peak_viewing = peak_viewing[['Household number', 'viewing time']].rename(columns={'viewing time': 'Peak Viewing Time'})

    # --- STEP 2: Genre Ratios ---
    combined_df = pd.merge(pivot_df, total_duration_df, on='Household number', how='left')
    genre_columns = [c for c in combined_df.columns if c not in ['Household number', 'Total Duration']]
    for col in genre_columns:
        combined_df[col] /= combined_df['Total Duration']
    combined_df[genre_columns] = combined_df[genre_columns].round(2)

    # --- STEP 3: Recording, Metadata, and Merge ---
    df['Recording'] = pd.to_numeric(df['Recording'], errors='coerce')
    recording_df = df[df['Recording'] == 1].groupby('Household number')['Duration of session'].sum().reset_index(name='Total Recording')
    recording_df = pd.DataFrame({'Household number': df['Household number'].unique()}).merge(recording_df, on='Household number', how='left')
    recording_df['Total Recording'] = recording_df['Total Recording'].fillna(0).astype(int)
    meta_df = df[['Household number', 'Area', 'Age', 'Processing Weight']].drop_duplicates('Household number', keep='last')

    combined = combined_df.merge(peak_viewing, on='Household number', how='left') \
                          .merge(recording_df, on='Household number', how='left') \
                          .merge(meta_df, on='Household number', how='left') \
                          .drop_duplicates('Household number')

    # --- STEP 4: Feature Engineering ---
    combined['Leisure_group'] = combined.get('Entertainment', 0) + combined.get('Movies&Dramas', 0) + combined.get('Children', 0) + combined.get('Lifestyle', 0) + combined.get('Music', 0)
    combined['Factual_group'] = combined.get('News', 0) + combined.get('Documentaries', 0)
    combined['Sport_group'] = combined.get('Sport', 0)
    combined['VOD_group'] = combined.get('VOD', 0)

    def classify_recorder(x):
        return 1 if x >= 120 else 0.75 if x >= 90 else 0.5 if x >= 60 else 0.25 if x > 0 else 0
    combined['is_recorder'] = combined['Total Recording'].apply(classify_recorder)
    combined['log_total_duration'] = np.log1p(combined['Total Duration'])

    time_map = {'Early_morning': 0.0, 'Morning': 0.25, 'Afternoon': 0.5, 'Evening': 0.75, 'Midnight': 1.0}
    combined['peak_time_numeric'] = combined['Peak Viewing Time'].map(time_map)

    # --- STEP 5: Clustering ---
    scaler = SCALER_PATH
    kmeans = KMEANS_PATH

    features = ['Leisure_group', 'Factual_group', 'Sport_group', 'VOD_group']
    combined['cluster'] = kmeans.predict(scaler.transform(combined[features]))

    # --- STEP 6: Interpretation ---
    def weighted_mean(group, col):
        return (group[col] * group['Processing Weight']).sum() / group['Processing Weight'].sum()
    metrics = ['log_total_duration', 'is_recorder', 'peak_time_numeric']
    cluster_summary = combined.groupby('cluster').apply(lambda g: pd.Series({m: weighted_mean(g, m) for m in metrics})).round(2)

    def interpret(row):
        v = "High" if row['log_total_duration'] > 5.3 else "Low" if row['log_total_duration'] < 5.0 else "Moderate"
        r = "most recording" if row['is_recorder'] > 0.4 else "some recording" if row['is_recorder'] > 0.25 else "rarely record"
        t = "tend to watch later" if row['peak_time_numeric'] > 0.7 else "prefer earlier viewing" if row['peak_time_numeric'] < 0.6 else "evening tendency"
        return f"{v} viewing time, {r}, {t}"

    cluster_summary['Interpretation'] = cluster_summary.apply(interpret, axis=1)
    cluster_summary['estimated_minutes'] = np.exp(cluster_summary['log_total_duration']).round(0).astype(int)
    cluster_summary['estimated_hhmm'] = cluster_summary['estimated_minutes'].apply(lambda x: f"{x // 60}h {x % 60}m")
    name_map = {0: 'Factual-heavy', 1: 'VOD-focused', 2: 'Entertainment-heavy', 3: 'Sport-focused'}
    cluster_summary = cluster_summary.reset_index()
    cluster_summary['Cluster Name'] = cluster_summary['cluster'].map(name_map)
    cluster_summary = cluster_summary[[
        'cluster', 'Cluster Name', 'estimated_hhmm', 'estimated_minutes',
        'log_total_duration', 'is_recorder', 'peak_time_numeric', 'Interpretation'
    ]]

    # --- STEP 7: Weighted Share Report ---
    long_term = {0: 19.3, 1: 39.9, 2: 37.2, 3: 3.6}
    audience = combined.groupby('cluster')['Processing Weight'].sum()
    total_audience = audience.sum()
    summary = pd.DataFrame({'Cluster': audience.index, 'Weighted Audience Size': audience.values})
    summary['% of Total'] = (summary['Weighted Audience Size'] / total_audience * 100).round(1)
    summary['Deviation from Avg'] = summary.apply(lambda r: f"{round(r['% of Total'] - long_term[r['Cluster']], 1):+}%", axis=1)
    summary['Cluster'] = summary['Cluster'].map(name_map)
    summary['Weighted Audience Size (M)'] = (summary['Weighted Audience Size'] / 1_000_000).round(1)
    summary = summary[['Cluster', 'Weighted Audience Size (M)', '% of Total', 'Deviation from Avg']]

    # --- STEP 8: Top Areas by Cluster ---
    top_areas = combined.groupby(['cluster', 'Area'])['Processing Weight'].sum().reset_index(name='Weighted Audience')
    top_areas = top_areas.sort_values(['cluster', 'Weighted Audience'], ascending=[True, False])
    top_areas['Rank'] = top_areas.groupby('cluster').cumcount() + 1
    top3 = top_areas[top_areas['Rank'] <= 3]
    top3['Cluster Name'] = top3['cluster'].map(name_map)
    top3['Audience (M)'] = (top3['Weighted Audience'] / 1_000_000).round(1)
    pivot = top3.pivot(index='Cluster Name', columns='Rank', values=['Area', 'Audience (M)'])
    pivot.columns = [f"{a} {b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()
    pivot = pivot.rename(columns={
        'Area 1': 'Top 1 Area', 'Audience (M) 1': 'Top 1 Audience (M)',
        'Area 2': 'Top 2 Area', 'Audience (M) 2': 'Top 2 Audience (M)',
        'Area 3': 'Top 3 Area', 'Audience (M) 3': 'Top 3 Audience (M)',
    })
    pivot = pivot[[
        'Cluster Name', 'Top 1 Area', 'Top 1 Audience (M)',
        'Top 2 Area', 'Top 2 Audience (M)', 'Top 3 Area', 'Top 3 Audience (M)'
    ]]

    return cluster_summary, summary, pivot



def plot_weighted_audience_bar(summary_df):
    """
    Creates and returns a bar chart of weighted audience size by cluster.

    Parameters:
        summary_df (DataFrame): A DataFrame containing at least 'Cluster' and 'Weighted Audience Size (M)' columns.

    Returns:
        fig (matplotlib.figure.Figure): The bar chart figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(summary_df['Cluster'], summary_df['Weighted Audience Size (M)'])

    # Annotate each bar with its value
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.2,
            f'{height:.1f}M',
            ha='center'
        )

    ax.set_title('Weighted Audience Size by Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Audience Size (M)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()

    return fig



def plot_deviation_from_average(summary_df):
    """
    Creates and returns a bar chart showing deviation from average audience share for each cluster.

    Parameters:
        summary_df (DataFrame): DataFrame with 'Cluster' and 'Deviation from Avg' columns.
                                'Deviation from Avg' should be in percentage string format (e.g., '+2.4%').

    Returns:
        fig (matplotlib.figure.Figure): The bar chart figure.
    """
    # Clean and convert deviation values
    summary_df['Deviation from Avg'] = summary_df['Deviation from Avg'].replace('[%]', '', regex=True).astype(float)

    clusters = summary_df['Cluster']
    deviation = summary_df['Deviation from Avg']
    colors = ['green' if val > 0 else 'red' for val in deviation]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(clusters, deviation, color=colors)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (0.3 if height >= 0 else -0.4),
            f'{height:+.1f}%',
            ha='center',
            va='bottom' if height >= 0 else 'top'
        )

    ax.set_title('Deviation from Average Audience Share by Cluster')
    ax.set_ylabel('Deviation (%)')
    ax.axhline(0, color='black', linewidth=1)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    fig.tight_layout()

    return fig

