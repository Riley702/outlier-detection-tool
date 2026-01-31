import pandas as pd

from outlier_detection_tool import detect_outliers, summarize_outliers


def test_detect_outliers_basic(tmp_path):
    data = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100], "y": [2, 4, 6, 8, 10, 200]})
    test_file = tmp_path / "data.csv"
    data.to_csv(test_file, index=False)

    result = detect_outliers(str(test_file), threshold=0.5)

    assert "cooks_distance" in result.columns
    assert "outlier" in result.columns
    assert bool(result.loc[result.index[-1], "outlier"]) is True


def test_summarize_outliers():
    data = pd.DataFrame(
        {
            "cooks_distance": [0.001, 0.8],
            "outlier": [False, True],
        }
    )

    summary = summarize_outliers(data)
    assert summary["total_points"] == 2
    assert summary["outliers"] == 1
    assert summary["non_outliers"] == 1


def test_missing_file(tmp_path):
    missing = tmp_path / "nope.csv"
    try:
        detect_outliers(str(missing))
        raise AssertionError("Expected FileNotFoundError")
    except FileNotFoundError:
        pass


def test_invalid_columns(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    p = tmp_path / "bad.csv"
    df.to_csv(p, index=False)

    try:
        detect_outliers(str(p))
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass
