#!/usr/bin/env python3
"""Normalize local model predictions into the paired CDP input schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bjd_cdp.predictions import normalize_prediction_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="One or more local prediction CSV files")
    parser.add_argument("--output", type=Path, required=True, help="Local aligned-correctness CSV")
    parser.add_argument(
        "--incomplete-policy",
        choices=("error", "drop"),
        default="error",
        help="Fail on incomplete four-condition samples, or drop them as complete sample groups",
    )
    parser.add_argument(
        "--exclusion-log",
        type=Path,
        default=None,
        help="CSV for locally recording dropped sample groups",
    )
    args = parser.parse_args()

    frames = [pd.read_csv(path, dtype={"key": "string"}) for path in args.input]
    normalized, exclusions = normalize_prediction_frames(
        frames,
        incomplete_policy=args.incomplete_policy,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(args.output, index=False, encoding="utf-8")

    if args.exclusion_log is not None:
        args.exclusion_log.parent.mkdir(parents=True, exist_ok=True)
        exclusions.to_csv(args.exclusion_log, index=False, encoding="utf-8")
    elif not exclusions.empty:
        raise ValueError("--exclusion-log is required when incomplete samples are dropped")

    print(f"Wrote {len(normalized)} paired condition rows to {args.output.resolve()}")
    print(f"Excluded sample groups: {len(exclusions)}")


if __name__ == "__main__":
    main()
