"""Outlier Detection Tool.

Primary API:
- detect_outliers: read a CSV with x,y and compute Cook's distance + outlier flag.
- summarize_outliers: summarize results.

"""

from .detect_outliers import detect_outliers, summarize_outliers

__all__ = ["detect_outliers", "summarize_outliers"]
