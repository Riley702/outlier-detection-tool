# Outlier Detection Tool

The **Outlier Detection Tool** is a robust Python package designed for data scientists, statisticians, and analysts who need a precise and efficient method for identifying anomalies in two-dimensional datasets. Utilizing the statistical rigor of Cook's Distance, this tool ingests (x, y) data from CSV files, performs regression diagnostics, and flags influential data points that deviate significantly from the general trend.

---

## Purpose and Motivation

In data-driven domains, identifying and addressing outliers is a critical step in ensuring the integrity and reliability of analytical models. Outliers can distort trends, bias predictions, and lead to incorrect conclusions. However, detecting such anomalies is not always straightforward, especially in complex datasets where simple methods like standard deviation thresholds may fail.

The **Outlier Detection Tool**, developed by **Yisong Chen**, addresses this challenge by leveraging Cook's Distance, a measure specifically designed to quantify the influence of individual data points in regression models. By automating this advanced statistical approach, this tool empowers professionals to:
- Identify data points that disproportionately affect regression outcomes.
- Quantify the degree of influence for each observation.
- Make informed decisions about whether to retain or exclude anomalous data.

The tool bridges the gap between theoretical rigor and practical application, making it an essential component for any professional engaged in data preprocessing or statistical modeling.

---

## Key Features

- **Cook's Distance Calculation**: Precisely measures the influence of each data point on regression results.
- **Automated Outlier Flagging**: Flags data points exceeding a user-defined threshold.
- **Customizable Threshold**: Allows fine-tuning of the sensitivity for outlier detection.
- **Summary Statistics**: Provides detailed insights, including the total number of outliers, their proportion, and their impact on the dataset.
- **Seamless Integration**: Ingests CSV files with (x, y) data and outputs enhanced datasets with anomaly flags and diagnostic metrics.
- **Error Handling**: Robust validation for missing files, malformed data, or incomplete input structures.

---

## Installation

To install the package, use `pip`:
```bash
pip install outlier-detection-tool
