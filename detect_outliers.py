import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

def detect_outliers(file_path, threshold=0.5, output_file="output_with_outliers.csv"):
    """
    Detect outliers in (x, y) data using Cook's distance.

    Args:
        file_path (str): Path to the CSV file containing 'x' and 'y' data.
        threshold (float): Threshold for Cook's distance to flag outliers.
        output_file (str): Path to save the processed file with outliers flagged.

    Returns:
        pd.DataFrame: DataFrame with calculated Cook's distance and outlier flags.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    data = pd.read_csv(file_path)

    if 'x' not in data.columns or 'y' not in data.columns:
        raise ValueError("The input CSV file must contain 'x' and 'y' columns.")

    # Prepare the data for regression
    X = sm.add_constant(data['x'])
    y = data['y']

    model = sm.OLS(y, X).fit()

    # Calculate Cook's distance and influence metrics
    influence = model.get_influence()
    cooks = influence.cooks_distance[0]

    # Add results to the DataFrame
    data['cooks_distance'] = cooks
    data['outlier'] = data['cooks_distance'] > threshold

    # Optional: Save output to a CSV file
    data.to_csv(output_file, index=False)

    return data

def summarize_outliers(data):
    """
    Summarize the outlier detection results.

    Args:
        data (pd.DataFrame): DataFrame containing the outlier flags.

    Returns:
        dict: Summary statistics including count of outliers and non-outliers.
    """
    total_points = len(data)
    outlier_count = data['outlier'].sum()
    non_outlier_count = total_points - outlier_count

    return {
        "total_points": total_points,
        "outliers": outlier_count,
        "non_outliers": non_outlier_count,
        "outlier_percentage": (outlier_count / total_points) * 100,
    }

if __name__ == "__main__":
    input_file = "data.csv"
    output_file = "output_with_outliers.csv"

    # Detect outliers and save results
    try:
        processed_data = detect_outliers(input_file, threshold=0.5, output_file=output_file)
        summary = summarize_outliers(processed_data)

        print("Outlier Detection Completed")
        print("Summary Statistics:")
        for key, value in summary.items():
            print(f"{key.capitalize()}: {value}")

    except Exception as e:
        print(f"Error: {e}")
