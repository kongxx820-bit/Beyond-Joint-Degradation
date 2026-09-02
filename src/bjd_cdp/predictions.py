"""Prediction parsing and matched-table construction utilities.

The functions in this module operate on locally generated inference records.
They contain no paper data or expected experimental values.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping

import pandas as pd


GROUP_COLUMNS = ["model", "language", "pvd", "tri"]
PREDICTION_COLUMNS = {
    "key",
    "label",
    "pred",
    "T",
    "V",
    *GROUP_COLUMNS,
}
EXPECTED_CELLS = {(0, 0), (1, 0), (0, 1), (1, 1)}
CONDITION_BY_CELL = {
    (0, 0): "clean",
    (1, 0): "text",
    (0, 1): "vision",
    (1, 1): "joint",
}


def parse_binary_response(value: object) -> int | None:
    """Parse an exact binary classifier response.

    Explanations or ambiguous strings are rejected. The raw response should be
    retained by the caller for local auditing.
    """
    if value is None:
        return None
    text = str(value).strip().casefold()
    text = re.sub(r"^[`*_\s]+|[`*_\s]+$", "", text)
    text = text.rstrip(".。!！")
    text = re.sub(r"\s+", " ", text)
    if text in {"harmful", "hateful", "1"}:
        return 1
    if text in {"non-harmful", "non harmful", "non-hateful", "non hateful", "0"}:
        return 0
    return None


def stable_row_id(row: Mapping[str, object]) -> str:
    """Return a stable identifier for one model-condition inference row."""
    fields = ("key", "model", "language", "pvd", "tri", "T", "V")
    payload = "\0".join(str(row.get(field, "")) for field in fields).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_prediction_frames(
    frames: Iterable[pd.DataFrame],
    *,
    incomplete_policy: str = "error",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a paired correctness table and an exclusion log.

    ``incomplete_policy`` is ``error`` or ``drop``. Under ``drop``, every
    incomplete sample is removed as a whole within its analysis grid row and is
    written to the returned exclusion table.
    """
    if incomplete_policy not in {"error", "drop"}:
        raise ValueError("incomplete_policy must be 'error' or 'drop'")

    materialized = [frame.copy() for frame in frames]
    if not materialized:
        raise ValueError("No prediction tables were supplied")
    data = pd.concat(materialized, ignore_index=True)
    missing = sorted(PREDICTION_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")

    data["key"] = data["key"].astype("string").str.strip()
    if data["key"].isna().any() or data["key"].eq("").any():
        raise ValueError("key must be non-empty")

    for column in ("label", "pred", "T", "V"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    invalid = data[["label", "pred", "T", "V"]].isna().any(axis=1)
    invalid |= ~data["label"].isin([0, 1])
    invalid |= ~data["pred"].isin([0, 1])
    invalid |= ~data["T"].isin([0, 1])
    invalid |= ~data["V"].isin([0, 1])
    if invalid.any():
        raise ValueError(
            f"Found {int(invalid.sum())} rows with missing or non-binary label/pred/T/V values"
        )

    for column in ("label", "pred", "T", "V"):
        data[column] = data[column].astype(int)

    identity = GROUP_COLUMNS + ["key", "T", "V"]
    duplicates = data.duplicated(identity, keep=False)
    if duplicates.any():
        preview = data.loc[duplicates, identity].head(5).to_dict("records")
        raise ValueError(f"Duplicate sample-condition rows found: {preview}")

    sample_columns = GROUP_COLUMNS + ["key"]
    incomplete_keys: list[dict[str, object]] = []
    for sample_key, group in data.groupby(sample_columns, sort=False, dropna=False):
        observed = set(zip(group["T"], group["V"]))
        if observed != EXPECTED_CELLS:
            record = dict(zip(sample_columns, sample_key))
            record["observed_cells"] = ",".join(f"{t}{v}" for t, v in sorted(observed))
            incomplete_keys.append(record)

    exclusions = pd.DataFrame(incomplete_keys)
    if incomplete_keys and incomplete_policy == "error":
        raise ValueError(
            f"Found {len(incomplete_keys)} incomplete matched samples; rerun with "
            "--incomplete-policy drop only if this exclusion rule was prespecified"
        )
    if incomplete_keys:
        excluded_index = pd.MultiIndex.from_frame(exclusions[sample_columns])
        current_index = pd.MultiIndex.from_frame(data[sample_columns])
        data = data.loc[~current_index.isin(excluded_index)].copy()

    data["correct"] = data["label"].eq(data["pred"]).astype(int)
    data["condition"] = [CONDITION_BY_CELL[(t, v)] for t, v in zip(data["T"], data["V"])]
    ordered = [
        "key",
        "label",
        "pred",
        "correct",
        "condition",
        "T",
        "V",
        *GROUP_COLUMNS,
    ]
    remaining = [column for column in data.columns if column not in ordered]
    return data[ordered + remaining].sort_values(identity).reset_index(drop=True), exclusions
