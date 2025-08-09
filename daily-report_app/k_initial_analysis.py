import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


def assign_time_block(minutes):
    """
    Categorise total minutes since midnight into a time block.
    """
    if 120 <= minutes <= 419: return 'Early_morning'
    elif 420 <= minutes <= 719: return 'Morning'
    elif 720 <= minutes <= 1019: return 'Afternoon'
    elif 1020 <= minutes <= 1319: return 'Evening'
    elif 1320 <= minutes <= 1559: return 'Midnight'
    else: return 'Unknown'


def clearning (df):
    """
    Clean and prepare viewing session data.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'Genre', 'Duration of session', 'Start time of session'.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame with added time-related columns.
    """
        
    df.columns = df.columns.str.strip()
    df['Genre'] = df['Genre'].replace('Regional general entertainment', 'Entertainment')
    df = df[df['Genre'] != 'No programmes']

    # Clean and convert
    df['Duration of session'] = pd.to_numeric(df['Duration of session'], errors='coerce')
    df['Start time of session'] = pd.to_numeric(df['Start time of session'], errors='coerce')

    # Time features
    df['hour'] = df['Start time of session'] // 100
    df['minute'] = df['Start time of session'] % 100
    df['total_minutes'] = df['hour'] * 60 + df['minute']
    df['viewing time'] = df['total_minutes'].apply(assign_time_block)

    return df



def plot_unique_household_comparison(base_df, today_df, column='Household number'):
    """
    Creates a bar chart comparing unique household counts between two DataFrames.

    Parameters:
        base_df (pd.DataFrame): The base/reference dataset.
        today_df (pd.DataFrame): The current or comparison dataset.
        column (str): The name of the column containing household IDs (default: 'Household number').

    Returns:
        fig (matplotlib.figure.Figure): The bar chart figure.
    """
    unique_base = base_df[column].nunique()
    unique_today = today_df[column].nunique()

    labels = ['Base', 'Today']
    counts = [unique_base, unique_today]

    fig, ax = plt.subplots(figsize=(3, 3))
    bars = ax.bar(labels, counts, color=['#4C72B0', '#55A868'])
    ax.set_ylabel('Unique Household Count')
    ax.set_title('Comparison of Unique Household Counts')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{int(height)}',
                ha='center', va='bottom', fontsize=10)

    fig.tight_layout()
    return fig




def compare_households_by_age(base_df, today_df, age_column='Age', household_column='Household number'):
    """
    Compares unique household counts by age group between two DataFrames and returns the bar chart figure.

    Parameters:
        base_df (pd.DataFrame): The baseline dataset.
        today_df (pd.DataFrame): The comparison dataset (e.g., today's data).
        age_column (str): Name of the column containing age values.
        household_column (str): Name of the column containing household IDs.

    Returns:
        fig (matplotlib.figure.Figure): The bar chart figure comparing unique household counts by age group.
    """
    # Define bins and labels
    bins = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, float('inf')]
    labels = ['0–9', '10–19', '20–29', '30–39', '40–49',
              '50–59', '60–69', '70–79', '80–89', '90–99', '100+']

    # Ensure Age is numeric
    base_df[age_column] = pd.to_numeric(base_df[age_column], errors='coerce')
    today_df[age_column] = pd.to_numeric(today_df[age_column], errors='coerce')

    # Apply age binning
    base_df['AgeGroup'] = pd.cut(base_df[age_column], bins=bins, labels=labels, right=True)
    today_df['AgeGroup'] = pd.cut(today_df[age_column], bins=bins, labels=labels, right=True)

    # Count unique households per age group
    base_counts = base_df.groupby('AgeGroup')[household_column].nunique()
    today_counts = today_df.groupby('AgeGroup')[household_column].nunique()

    # Combine counts into one DataFrame
    comparison_df = pd.DataFrame({
        'Base': base_counts,
        'Today': today_counts
    }).fillna(0).astype(int)

    # Create and return the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_df.plot(kind='bar', ax=ax)
    ax.set_ylabel('Unique Household Count')
    ax.set_title('Comparison of Unique Households by Age Group')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend(title='DataFrame')
    fig.tight_layout()

    return fig



def compare_households_by_area(base_df, today_df, area_column='Area', household_column='Household number'):
    """
    Compares unique household counts by area between two DataFrames and returns the bar chart figure.

    Parameters:
        base_df (pd.DataFrame): The baseline dataset.
        today_df (pd.DataFrame): The comparison dataset (e.g., today's data).
        area_column (str): Name of the column containing area names.
        household_column (str): Name of the column containing household IDs.

    Returns:
        fig (matplotlib.figure.Figure): The bar chart figure comparing unique household counts by area.
    """
    # Group and count unique households by area
    base_area_counts = base_df.groupby(area_column)[household_column].nunique()
    today_area_counts = today_df.groupby(area_column)[household_column].nunique()

    # Combine into a single DataFrame
    area_comparison_df = pd.DataFrame({
        'Base': base_area_counts,
        'Today': today_area_counts
    }).fillna(0).astype(int)

    # Optional: sort by total count for better visual ordering
    area_comparison_df['Total'] = area_comparison_df['Base'] + area_comparison_df['Today']
    area_comparison_df = area_comparison_df.sort_values('Total', ascending=False).drop(columns='Total')

    # Create and return the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    area_comparison_df.plot(kind='bar', ax=ax)
    ax.set_ylabel('Unique Household Count')
    ax.set_title('Comparison of Unique Households by Area')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='DataFrame')
    fig.tight_layout()

    return fig
