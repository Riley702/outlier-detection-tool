# Outlier Detection Tool

## Overview
The **Outlier Detection Tool** is a powerful and efficient Python package designed for data scientists, statisticians, and analysts who require a precise method to detect anomalies in two-dimensional datasets. By leveraging the statistical rigor of **Cook's Distance**, this tool performs regression diagnostics on (x, y) data, identifies influential data points, and flags significant deviations from the general trend.

----

## Purpose and Motivation

Outliers can significantly impact statistical models, distort trends, and bias predictions. Addressing these anomalies is essential for ensuring the accuracy and reliability of data-driven insights. Traditional methods like standard deviation thresholds or IQR (Interquartile Range) often fall short in complex datasets where more nuanced techniques are required.

The **Outlier Detection Tool**, developed by **Yisong Chen**, provides a more sophisticated approach by utilizing **Cook's Distance**, a metric designed to quantify the influence of each observation in a regression model. This tool enables professionals to:

- Detect and quantify data points that disproportionately affect regression outcomes.
- Evaluate the statistical influence of each observation and determine its impact on overall model performance.
- Make informed decisions about whether to retain or exclude anomalous data.

By automating the use of Cook's Distance, the **Outlier Detection Tool** bridges the gap between theoretical rigor and practical application, making it indispensable for any professional working in data preprocessing, anomaly detection, or predictive modeling.

----

## Key Features

- **Cook's Distance Calculation**: Precisely measures the influence of each data point in regression models.
- **Automated Outlier Flagging**: Flags influential data points based on a user-defined threshold.
- **Customizable Sensitivity**: Allows fine-tuning of the outlier detection threshold to fit different datasets.
- **Comprehensive Summary Statistics**: Provides detailed insights, including the number of outliers detected, their proportion, and their influence on the dataset.
- **CSV File Support**: Ingests CSV files with (x, y) data and outputs enriched datasets with diagnostic metrics and anomaly flags.
- **Robust Error Handling**: Includes validation for missing files, malformed data, or incomplete input structures to prevent processing failures.
- **Flexible Data Cleaning**: Offers built-in utilities for removing duplicates, handling missing values, and filtering datasets based on custom criteria.
- **Advanced Data Processing**: Supports additional functionalities such as feature scaling, sorting, renaming columns, and computing correlation matrices.

----

## Installation

To install the **Outlier Detection Tool**, ensure that Python and `pip` are installed on your system. You can install the required dependencies using:

```bash
pip install -r requirements.txt
```

or manually install the key libraries:

```bash
pip install pandas numpy statsmodels
```

----

## Usage

### 1. Detecting Outliers
The primary function of the tool is to detect outliers in (x, y) datasets using Cook’s Distance. To run outlier detection on a CSV file, use:

```python
from detect_outliers import detect_outliers

file_path = "data.csv"
outlier_data = detect_outliers(file_path, threshold=0.5, output_file="output_with_outliers.csv")
```

### 2. Summarizing Outlier Results

```python
from detect_outliers import summarize_outliers

summary = summarize_outliers(outlier_data)
print(summary)
```

### 3. Cleaning Data

```python
from data_analysis import clean_data

cleaned_data = clean_data("data.csv", output_file="cleaned_data.csv")
```

### 4. Feature Scaling

```python
from data_analysis import scale_features

scaled_data = scale_features(cleaned_data, ["x", "y"])
```

### 5. Sorting Data

```python
from data_analysis import sort_data

sorted_data = sort_data(cleaned_data, column="x", ascending=True)
```

----

## Example Dataset
To test the tool, use a simple CSV file (`data.csv`) structured as follows:

```
x,y
1,2
2,4
3,6
4,8
5,10
100,200
```

This dataset contains five standard points and one extreme outlier (`100,200`). Running the tool will flag `100,200` as an influential data point.

----

## Contributing
We welcome contributions to enhance the functionality of the **Outlier Detection Tool**. If you would like to contribute:

1. Fork the repository.
2. Create a new feature branch.
3. Implement and test your changes.
4. Submit a pull request for review.

For major changes, please open an issue first to discuss your proposed modifications.

----

## License
This project is licensed under the **MIT License**.

----

## Author
**Yisong Chen**  
For inquiries or collaborations, please reach out via GitHub or email.

