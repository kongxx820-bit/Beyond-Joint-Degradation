#!/usr/bin/env python3
"""Run CDP on a locally generated aligned-correctness table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bjd_cdp.diagnosis import analyze_cdp, classify_cdp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Long-form CSV or single-CSV ZIP archive")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--floor", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    result = analyze_cdp(frame, floor_threshold=args.floor, alpha=args.alpha)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "cdp_results.csv", index=False)

    summary = (
        result["cdp_status"]
        .value_counts()
        .reindex(["confirmed", "unsupported", "floor-limited"], fill_value=0)
        .rename_axis("cdp_status")
        .reset_index(name="count")
    )
    summary["percentage"] = 100.0 * summary["count"] / len(result)
    summary.to_csv(args.output_dir / "cdp_status_summary.csv", index=False)

    sensitivity_rows = []
    for threshold in (0.03, 0.05, 0.10):
        statuses = result.apply(
            lambda row: classify_cdp(
                float(row["A_text"]),
                float(row["A_vision"]),
                float(row["A_joint"]),
                float(row["beta_TV"]),
                float(row["q_TV"]),
                floor_threshold=threshold,
                alpha=args.alpha,
            ),
            axis=1,
        )
        counts = statuses.value_counts()
        sensitivity_rows.append(
            {
                "floor_threshold": threshold,
                "confirmed": int(counts.get("confirmed", 0)),
                "unsupported": int(counts.get("unsupported", 0)),
                "floor_limited": int(counts.get("floor-limited", 0)),
            }
        )
    pd.DataFrame(sensitivity_rows).to_csv(args.output_dir / "floor_threshold_sensitivity.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

