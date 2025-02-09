import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def clean_data(file_path, output_file="cleaned_data.csv"):
    """
    Cleans the input CSV data by removing duplicates and handling missing values.

    Args:
        file_path (str): Path to the input CSV file.
        output_file (str): Path to save the cleaned data.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    logging.info(f"Starting data cleaning for file: {file_path}")

    try:
        data = pd.read_csv(file_path)
        logging.info("Data successfully loaded.")
        
        # Remove duplicates
        data = data.drop_duplicates()
        logging.info("Duplicates removed.")
        
        # Handle missing values
        data = data.dropna()
        logging.info("Missing values removed.")

        # Save cleaned data
        if output_file:
            data.to_csv(output_file, index=False)
            logging.info(f"Cleaned data saved to: {output_file}")

        return data

    except Exception as e:
        logging.error(f"Error occurred during data cleaning: {e}")
        raise

def filter_data(data, column, threshold):
    """
    Filters data based on a threshold value for a specific column.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        column (str): Column name to apply the threshold.
        threshold (float): Threshold value to filter the data.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    logging.info(f"Filtering data where {column} >= {threshold}")
    try:
        filtered_data = data[data[column] >= threshold]
        logging.info(f"Filtered {len(data) - len(filtered_data)} rows below the threshold.")
        return filtered_data
    except Exception as e:
        logging.error(f"Error occurred during data filtering: {e}")
        raise

def convert_column_to_numeric(data, column):
    """
    Converts a specified column to numeric format, handling errors gracefully.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        column (str): Column name to convert to numeric.

    Returns:
        pd.DataFrame: DataFrame with the column converted to numeric.
    """
    logging.info(f"Converting column {column} to numeric format...")
    try:
        data[column] = pd.to_numeric(data[column], errors='coerce')
        logging.info(f"Column {column} converted to numeric successfully.")
        return data
    except Exception as e:
        logging.error(f"Error occurred during column conversion: {e}")
        raise
