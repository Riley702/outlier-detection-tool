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

    try:


    except Exception as e:
        logging.error(f"Error while calculating statistics: {e}")
        raise

