import os
import pandas as pd
from outlier_detection_tool.detect_outliers import detect_outliers, summarize_outliers

def test_detect_outliers_basic():
    """
    Test the basic functionality of detecting outliers in a small dataset.
    """
    # Create a sample dataset
    data = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100], "y": [2, 4, 6, 8, 10, 200]})
    test_file = "test_data.csv"
    data.to_csv(test_file, index=False)

    # Run outlier detection
    result = detect_outliers(test_file, threshold=0.5)

    # Assertions
    assert 'cooks_distance' in result.columns, "Cook's distance column is missing."
    assert 'outlier' in result.columns, "Outlier flag column is missing."
    assert result['outlier'].iloc[-1], "The last row should be flagged as an outlier."
    assert not result['outlier'].iloc[:-1].any(), "Non-outlier rows incorrectly flagged."

    # Clean up
    os.remove(test_file)
    print("test_detect_outliers_basic passed.")


def test_summarize_outliers():
    """
    Test the summarize_outliers function for correct statistical output.
    """
    data = pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 100],
        "y": [2, 4, 6, 8, 10, 200],
        "cooks_distance": [0.001, 0.002, 0.003, 0.004, 0.005, 0.8],
        "outlier": [False, False, False, False, False, True],
    })

    summary = summarize_outliers(data)

    assert summary["total_points"] == 6, "Total points calculation is incorrect."
    assert summary["outliers"] == 1, "Outlier count calculation is incorrect."
    assert summary["non_outliers"] == 5, "Non-outlier count calculation is incorrect."
    assert summary["outlier_percentage"] == (1 / 6) * 100, "Outlier percentage is incorrect."

    print("test_summarize_outliers passed.")


def test_missing_file():
    """
    Test behavior when a non-existent file is provided.
    """
    try:
        detect_outliers("non_existent_file.csv")
    except FileNotFoundError as e:
        assert str(e).startswith("The file"), "FileNotFoundError not raised correctly."
        print("test_missing_file passed.")
    else:
        raise AssertionError("FileNotFoundError was not raised as expected.")


def test_invalid_columns():
    """
    Test behavior when input data does not contain required columns.
    """
    data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    test_file = "test_invalid_columns.csv"
    data.to_csv(test_file, index=False)

    try:
        detect_outliers(test_file)
    except ValueError as e:
        assert str(e).startswith("The input CSV file must contain"), "ValueError not raised correctly for missing columns."
        print("test_invalid_columns passed.")
    else:
        raise AssertionError("ValueError was not raised as expected.")

    os.remove(test_file)

def test_no_outliers_detected():
    """
    Test behavior when no outliers are present in the dataset.
    """
    # Create a dataset with no significant outliers
    data = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
    test_file = "test_no_outliers.csv"
    data.to_csv(test_file, index=False)

    # Run outlier detection
    result = detect_outliers(test_file, threshold=1.0)  # Set a high threshold to avoid outliers

    # Assertions
    assert 'cooks_distance' in result.columns, "Cook's distance column is missing."
    assert 'outlier' in result.columns, "Outlier flag column is missing."
    assert not result['outlier'].any(), "No rows should be flagged as outliers."

    # Clean up
    os.remove(test_file)
    print("test_no_outliers_detected passed.")



if __name__ == "__main__":
    test_detect_outliers_basic()
    test_summarize_outliers()
    test_missing_file()
    test_invalid_columns()
    print("All tests passed successfully.")
