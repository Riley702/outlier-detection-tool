from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutlierDetectionResult:
    """Convenience wrapper (optional)."""

    data: pd.DataFrame


def detect_outliers(
    file_path: str,
    threshold: float = 0.5,
    output_file: Optional[str] = None,
) -> pd.DataFrame:
    """Detect outliers in (x, y) CSV data using Cook's distance.

    The input CSV must contain columns `x` and `y`.

    Args:
        file_path: Path to CSV.
        threshold: Cook's distance threshold for flagging outliers.
        output_file: If provided, writes the enriched CSV to this path.

    Returns:
        DataFrame with `cooks_distance` (float) and `outlier` (bool).
    """
    logger.info("Starting outlier detection for file: %s", file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    data = pd.read_csv(file_path)

    # Be forgiving: if the CSV has exactly 2 columns but no headers, treat them as x,y.
    if "x" not in data.columns or "y" not in data.columns:
        if len(data.columns) == 2:
            data = data.copy()
            data.columns = ["x", "y"]
        else:
            raise ValueError("The input CSV file must contain 'x' and 'y' columns (or exactly 2 columns without headers).")

    # Prepare regression
    X = sm.add_constant(data["x"])
    y = data["y"]

    model = sm.OLS(y, X).fit()
    influence = model.get_influence()
    cooks = influence.cooks_distance[0]

    data = data.copy()
    data["cooks_distance"] = cooks
    data["outlier"] = data["cooks_distance"] > float(threshold)

    if output_file:
        data.to_csv(output_file, index=False)
        logger.info("Wrote results to: %s", output_file)

    return data


def summarize_outliers(data: pd.DataFrame) -> dict:
    """Summarize outlier detection results."""
    if "outlier" not in data.columns:
        raise ValueError("DataFrame must contain an 'outlier' column.")

    total_points = int(len(data))
    outlier_count = int(data["outlier"].sum())
    non_outlier_count = total_points - outlier_count

    return {
        "total_points": total_points,
        "outliers": outlier_count,
        "non_outliers": non_outlier_count,
        "outlier_percentage": (outlier_count / total_points) * 100 if total_points else 0.0,
    }
