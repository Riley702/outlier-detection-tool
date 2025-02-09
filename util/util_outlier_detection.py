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

def rename_columns(data, column_mappings):
    """
    Rename columns in the DataFrame based on a given mapping.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        column_mappings (dict): Dictionary mapping old column names to new names.

    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """
    logging.info(f"Renaming columns: {column_mappings}")
    try:
        data = data.rename(columns=column_mappings)
        logging.info("Columns renamed successfully.")
        return data
    except Exception as e:
        logging.error(f"Error occurred during column renaming: {e}")
        raise

def sort_data(data, column, ascending=True):
    """
    Sort the DataFrame based on a specified column.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        column (str): Column name to sort by.
        ascending (bool): Whether to sort in ascending order (default=True).

    Returns:
        pd.DataFrame: Sorted DataFrame.
    """
    logging.info(f"Sorting data by column '{column}', ascending={ascending}")
    try:
        sorted_data = data.sort_values(by=column, ascending=ascending)
        logging.info("Data sorted successfully.")
        return sorted_data
    except Exception as e:
        logging.error(f"Error occurred during sorting: {e}")
        raise

def compute_unique_values(data, column):
    """
    Compute the number of unique values in a specified column.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        column (str): Column name to count unique values.

    Returns:
        int: Number of unique values in the column.
    """
    logging.info(f"Computing unique values in column '{column}'")
    try:
        unique_count = data[column].nunique()
        logging.info(f"Column '{column}' has {unique_count} unique values.")
        return unique_count
    except Exception as e:
        logging.error(f"Error occurred while computing unique values: {e}")
        raise
