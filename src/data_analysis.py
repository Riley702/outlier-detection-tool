import pandas as pd
import logging

# Configure logg
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

if __name__ == "__main__":
    # Example usage
    data = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [2, 4, 6, 8, 10]
    })

    column = 'x'
    stats = calculate_statistics(data, column)
    print(f"Statistics for '{column}': {stats}")
    
    missing_values = detect_missing_values(data)
    print(f"Missing values per column: {missing_values}")
    
    normalized_data = normalize_column(data, column)
    print(f"Normalized '{column}' column:")
    print(normalized_data)
