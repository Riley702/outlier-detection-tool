# Outlier Detection Tool

Detect outliers in 2D (x, y) datasets using **Cook's distance**.

## Install (editable / dev)

```bash
python3 -m pip install -e .[dev]
```

## Quickstart

```python
from outlier_detection_tool import detect_outliers, summarize_outliers

df = detect_outliers("data/data.csv", threshold=0.5)
print(summarize_outliers(df))
```

## CLI

```bash
outlier-detect --input data/data.csv --threshold 0.5 --output output_with_outliers.csv
```

## Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```
