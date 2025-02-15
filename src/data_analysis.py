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

if __name__ == "__main__":
    # Example usage
    data = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [2, 4, 6, 8, 10]
    })

    column = 'x'
    stats = calculate_statistics(data, column)
    print(f"Statistics for '{column}': {stats}")
