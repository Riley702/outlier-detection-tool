import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def calculate_statistics(data, column):
    """
    Calculate basic statistics (mean, median, standard deviation) for a specified column.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        column (str): Name of the column to calculate statistics for.

    Returns:
        dict: Dictionary containing mean, median, and standard deviation.
    """
    if column not in data.columns:
        logging.error(f"Column '{column}' not found in the data.")
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    logging.info(f"Calculating statistics for column: {column}")

    try:
        mean = data[column].mean()
        median = data[column].median()
        std_dev = data[column].std()

        stats = {
            "mean": mean,
            "median": median,
            "std_dev": std_dev
        }

        logging.info(f"Statistics for column '{column}': {stats}")
        return stats

    except Exception as e:
        logging.error(f"Error while calculating statistics: {e}")
        raise

def detect_missing_values(data):
    """
    Detect missing values in each column of the DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing the data.

    Returns:
        dict: Dictionary containing the count of missing values per column.
    """
    logging.info("Detecting missing values in the dataset...")
    try:
        missing_values = data.isnull().sum().to_dict()
        logging.info(f"Missing values per column: {missing_values}")
        return missing_values
    except Exception as e:
        logging.error(f"Error while detecting missing values: {e}")
        raise

def normalize_column(data, column):
    """
    Normalize a specified column using min-max normalization.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        column (str): Name of the column to normalize.

    Returns:
        pd.DataFrame: DataFrame with the normalized column.
    """
    if column not in data.columns:
        logging.error(f"Column '{column}' not found in the data.")
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    logging.info(f"Normalizing column: {column}")
    try:
        min_val = data[column].min()
        max_val = data[column].max()
        data[column] = (data[column] - min_val) / (max_val - min_val)
        logging.info(f"Column '{column}' normalized successfully.")
        return data
    except Exception as e:
        logging.error(f"Error while normalizing column '{column}': {e}")
        raise
def replace_missing_values(data, column, method="mean"):
    """
    Replace missing values in a specified column using mean, median, or mode.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        column (str): Name of the column to process.
        method (str): Strategy to replace missing values (options: "mean", "median", "mode").

    Returns:
        pd.DataFrame: DataFrame with missing values replaced.
    """
    if column not in data.columns:
        logging.error(f"Column '{column}' not found in the data.")
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    logging.info(f"Replacing missing values in column: {column} using {method} method.")

    try:
        if method == "mean":
            data[column].fillna(data[column].mean(), inplace=True)
        elif method == "median":
            data[column].fillna(data[column].median(), inplace=True)
        elif method == "mode":
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:
            raise ValueError("Method should be 'mean', 'median', or 'mode'.")

        logging.info(f"Missing values in column '{column}' replaced successfully.")
        return data
    except Exception as e:
        logging.error(f"Error while replacing missing values in column '{column}': {e}")
        raise

def compute_correlation_matrix(data):
    """
    Compute the correlation matrix for numerical columns in the DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing the data.

    Returns:
        pd.DataFrame: Correlation matrix.
    """
    logging.info("Computing correlation matrix for numerical columns...")
    try:
        correlation_matrix = data.corr()
        logging.info("Correlation matrix computed successfully.")
        return correlation_matrix
    except Exception as e:
        logging.error(f"Error while computing correlation matrix: {e}")
        raise
