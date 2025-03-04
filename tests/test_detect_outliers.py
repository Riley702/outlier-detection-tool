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

def test_outlier_removal_effect():
    """
    Test whether removing outliers changes the dataset size correctly.
    """
    data = pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 100],
        "y": [2, 4, 6, 8, 10, 200],
        "outlier": [False, False, False, False, False, True]
    })
    
    data_filtered = data[~data['outlier']]
    
    assert len(data_filtered) == 5, "Outlier removal did not adjust dataset size correctly."
    print("test_outlier_removal_effect passed.")

def test_outlier_summary_consistency():
    """
    Test if summary statistics remain consistent before and after removing outliers.
    """
    data = pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 100],
        "y": [2, 4, 6, 8, 10, 200],
        "outlier": [False, False, False, False, False, True]
    })
    
    summary_before = summarize_outliers(data)
    data_filtered = data[~data['outlier']]
    summary_after = summarize_outliers(data_filtered)
    
    assert summary_before["outliers"] == 1, "Initial summary miscounts outliers."
    assert summary_after["outliers"] == 0, "Filtered summary should not contain outliers."
    print("test_outlier_summary_consistency passed.")

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

def test_outlier_percentage_consistency():
    """
    Test if outlier percentage calculation remains consistent after filtering.
    """
    data = pd.DataFrame({
        "x": range(1, 21),
        "y": range(2, 42, 2),
        "outlier": [False] * 18 + [True, True]
    })

    summary_before = summarize_outliers(data)
    filtered_data = data[~data["outlier"]]
    summary_after = summarize_outliers(filtered_data)

    assert summary_before["outlier_percentage"] > summary_after["outlier_percentage"], "Outlier percentage should decrease after filtering."
    print("test_outlier_percentage_consistency passed.")

def test_large_dataset_performance():
    """
    Test the performance of the outlier detection function with a large dataset.
    """
    data = pd.DataFrame({
        "x": range(1, 10001),
        "y": range(2, 20002, 2)
    })

    test_file = "large_test_data.csv"
    data.to_csv(test_file, index=False)

    try:
        result = detect_outliers(test_file, threshold=0.5)
        assert len(result) == 10000, "The result should contain the same number of rows as the input."
        print("test_large_dataset_performance passed.")
    finally:
        os.remove(test_file)


if __name__ == "__main__":
    test_detect_outliers_basic()
    test_summarize_outliers()
    test_missing_file()
    test_invalid_columns()
    test_no_outliers_detected()
    test_outlier_removal_effect()
    test_outlier_summary_consistency()
    print("All tests passed successfully.")
