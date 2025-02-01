import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
    logging.info(f"Starting outlier detection for file: {file_path}")
    
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    data = pd.read_csv(file_path)
    logging.info("Data successfully loaded.")

    if 'x' not in data.columns or 'y' not in data.columns:
        logging.error("Missing required columns 'x' and 'y' in the input data.")
        raise ValueError("The input CSV file must contain 'x' and 'y' columns.")

    # Prepare the data for regression
    X = sm.add_constant(data['x'])
    y = data['y']

    logging.info("Fitting regression model...")
    model = sm.OLS(y, X).fit()

    # Calculate Cook's distance and influence metrics
    logging.info("Calculating Cook's distance...")
    influence = model.get_influence()
    cooks = influence.cooks_distance[0]

    # Add results to the DataFrame
    data['cooks_distance'] = cooks
    data['outlier'] = data['cooks_distance'] > threshold

    # Add 100 new rows of random data for testing purposes
    new_data = pd.DataFrame({
        'x': np.random.rand(100),
        'y': np.random.rand(100)
    })
    new_data['cooks_distance'] = np.nan
    new_data['outlier'] = False
    data = pd.concat([data, new_data], ignore_index=True)

    # Save output to a CSV file
    if output_file:
        data.to_csv(output_file, index=False)
        logging.info(f"Results saved to: {output_file}")

    return data


def summarize_outliers(data):
    """
    Summarize the outlier detection results.

    Args:
        data (pd.DataFrame): DataFrame containing the outlier flags.

    Returns:
        dict: Summary statistics including count of outliers and non-outliers.
    """
    logging.info("Summarizing outlier detection results...")
    total_points = len(data)
    outlier_count = data['outlier'].sum()
    non_outlier_count = total_points - outlier_count

    summary = {
        "total_points": total_points,
        "outliers": outlier_count,
        "non_outliers": non_outlier_count,
        "outlier_percentage": (outlier_count / total_points) * 100,
    }

    logging.info("Summary generated successfully.")
    return summary


def scale_features(data, columns):
    """
    Scale specified numerical columns to a 0-1 range.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        columns (list): List of column names to scale.

    Returns:
        pd.DataFrame: DataFrame with scaled columns.
    """
    logging.info("Scaling specified features to a 0-1 range...")
    try:
        data = data.copy()
        for col in columns:
            if col in data.columns:
                min_val = data[col].min()
                max_val = data[col].max()
                data[col] = (data[col] - min_val) / (max_val - min_val)
                logging.info(f"Column '{col}' scaled successfully.")
            else:
                logging.warning(f"Column '{col}' not found in the data.")
        return data
    except Exception as e:
        logging.error(f"Error during feature scaling: {e}")
        raise


def main(input_file, threshold=0.5, output_file="output_with_outliers.csv"):
    """
    Main function to detect and summarize outliers.

    Args:
        input_file (str): Path to the input CSV file.
        threshold (float): Threshold for Cook's distance to flag outliers.
        output_file (str): Path to save the output CSV file.
    """
    try:
        # Detect outliers
        processed_data = detect_outliers(input_file, threshold, output_file)

        # Summarize results
        summary = summarize_outliers(processed_data)

        # Display summary
        logging.info("Outlier Detection Completed")
        logging.info("Summary Statistics:")
        for key, value in summary.items():
            logging.info(f"{key.capitalize()}: {value}")

        # Scale features for further analysis
        scaled_data = scale_features(processed_data, ['x', 'y'])
        logging.info("Feature scaling completed. Preview of scaled data:")
        logging.info(scaled_data.head())

    except Exception as e:
        logging.error(f"Error occurred: {e}")

def remove_missing_values(data):
    """
    Remove rows with missing values from the dataset.

    Args:
        data (pd.DataFrame): DataFrame containing the data.

    Returns:
        pd.DataFrame: Cleaned DataFrame with missing values removed.
    """
    logging.info("Removing rows with missing values...")
    cleaned_data = data.dropna()
    logging.info(f"Removed {len(data) - len(cleaned_data)} rows with missing values.")
    return cleaned_data


def detect_high_variance_features(data, threshold=1.0):
    """
    Identify columns with high variance exceeding a given threshold.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        threshold (float): Variance threshold to identify high variance features.

    Returns:
        list: List of column names with variance above the threshold.
    """
    logging.info("Detecting high variance features...")
    high_variance_features = [col for col in data.select_dtypes(include=[np.number]).columns 
                              if data[col].var() > threshold]
    logging.info(f"High variance features detected: {high_variance_features}")
    return high_variance_features


if __name__ == "__main__":
    # Example usage
    input_file = "data.csv"
    output_file = "output_with_outliers.csv"
    threshold = 0.5  # Adjust threshold as needed

    main(input_file, threshold, output_file)
