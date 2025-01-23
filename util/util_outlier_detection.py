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
