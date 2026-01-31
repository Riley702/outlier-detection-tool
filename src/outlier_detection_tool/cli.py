from __future__ import annotations

import argparse
import logging

from .detect_outliers import detect_outliers


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Detect outliers using Cook's distance.")
    p.add_argument("--input", "-i", required=True, help="Input CSV path (must contain columns x,y)")
    p.add_argument("--threshold", "-t", type=float, default=0.5, help="Cook's distance threshold")
    p.add_argument("--output", "-o", default=None, help="Output CSV path")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable info logging")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    detect_outliers(args.input, threshold=args.threshold, output_file=args.output)
    return 0
