import pandas as pd
import numpy as np


def show_missing_values(df):
    """
    Identifies and calculates the volume and percentage of missing data across all columns.


    :param df: The DataFrame to be analyzed for missing (NaN) values.
    :type df: pandas.DataFrame
    :return: A summary DataFrame containing column names, missing counts, and their corresponding percentages.
    :rtype: pandas.DataFrame
    """
    missing_count = df.isna().sum()
    missing_percentage = round((df.isna().sum() / len(df)) * 100, 2)

    missing_data = pd.DataFrame(
        {
            "no_values_missing": missing_count,
            "percentage_values_missing": missing_percentage,
        }
    ).reset_index()

    missing_data = missing_data.rename(columns={"index": "column_name"})
    missing_data = missing_data.sort_values(
        by="percentage_values_missing", ascending=False
    )
    return missing_data


def calculate_precision(df, predicted_col, verified_col):
    """
    Calculates Precision for cases where only predicted positives were verified.


    :param df: The DataFrame containing verified samples.
    :type df: pandas.DataFrame
    :param predicted_col: The system's prediction column (expected to be all 1s here).
    :type predicted_col: str
    :param verified_col: The manual verification column.
    :type verified_col: str
    :return: Precision as a float (0 to 1).
    :rtype: float
    """

    tp = ((df[predicted_col] == 1) & (df[verified_col] == 1)).sum()

    fp = ((df[predicted_col] == 1) & (df[verified_col] == 0)).sum()

    if (tp + fp) == 0:
        return 0.0

    return round(float(tp / (tp + fp)), 2)
