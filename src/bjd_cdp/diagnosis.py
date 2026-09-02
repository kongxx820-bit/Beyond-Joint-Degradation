"""Cross-modal Diagnosis Protocol (CDP).

The implementation follows the camera-ready protocol:

* matched clean, text-only, vision-only, and joint conditions;
* JAS = min(A_text, A_vision) - A_joint;
* logistic correctness model with sample-clustered sandwich covariance;
* Benjamini-Hochberg correction within the supplied analysis grid; and
* a floor-first three-way status rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm


GROUP_COLUMNS = ["model", "language", "pvd", "tri"]
REQUIRED_COLUMNS = {
    "key",
    "correct",
    "T",
    "V",
    *GROUP_COLUMNS,
}


@dataclass(frozen=True)
class InteractionFit:
    beta_tv: float
    se_tv: float
    p_tv: float
    n_observations: int
    n_samples: int


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values, preserving NaNs."""
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    valid_values = values[valid]
    if valid_values.size == 0:
        return adjusted

    order = np.argsort(valid_values)
    ranked = valid_values[order]
    q_ranked = ranked * ranked.size / np.arange(1, ranked.size + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)

    restored = np.empty_like(q_ranked)
    restored[order] = q_ranked
    adjusted[valid] = restored
    return adjusted


def _validate(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    for column in ("correct", "T", "V"):
        values = set(pd.to_numeric(frame[column], errors="raise").unique())
        if not values.issubset({0, 1}):
            raise ValueError(f"{column} must be binary; found {sorted(values)}")
    duplicated = frame.duplicated(GROUP_COLUMNS + ["key", "T", "V"])
    if duplicated.any():
        raise ValueError("Each sample may appear at most once per grid row and condition")


def _cell_accuracy(frame: pd.DataFrame, text: int, vision: int) -> float:
    cell = frame[(frame["T"] == text) & (frame["V"] == vision)]
    if cell.empty:
        raise ValueError(f"Missing condition T={text}, V={vision}")
    return float(cell["correct"].mean())


def fit_interaction(frame: pd.DataFrame) -> InteractionFit:
    """Fit the saturated 2x2 logistic interaction model.

    Coefficients are obtained from the four cell logits. Standard errors use a
    cluster-robust sandwich covariance estimator with the source-sample ``key``
    as the cluster identifier. The finite-sample correction matches the common
    ``G/(G-1) * (N-1)/(N-K)`` correction for ``G`` clusters, ``N`` observations,
    and ``K`` fitted coefficients.
    """
    text = frame["T"].to_numpy(dtype=float)
    vision = frame["V"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(frame)), text, vision, text * vision])

    probabilities = {
        (0, 0): _cell_accuracy(frame, 0, 0),
        (1, 0): _cell_accuracy(frame, 1, 0),
        (0, 1): _cell_accuracy(frame, 0, 1),
        (1, 1): _cell_accuracy(frame, 1, 1),
    }
    if any(value <= 0.0 or value >= 1.0 for value in probabilities.values()):
        raise ValueError("A 0/1 cell accuracy causes complete separation; report this grid row separately")

    cell_logits = {cell: float(logit(value)) for cell, value in probabilities.items()}
    beta = np.array(
        [
            cell_logits[(0, 0)],
            cell_logits[(1, 0)] - cell_logits[(0, 0)],
            cell_logits[(0, 1)] - cell_logits[(0, 0)],
            cell_logits[(1, 1)]
            - cell_logits[(1, 0)]
            - cell_logits[(0, 1)]
            + cell_logits[(0, 0)],
        ],
        dtype=float,
    )

    fitted = expit(design @ beta)
    weights = fitted * (1.0 - fitted)
    bread = np.linalg.inv(design.T @ (design * weights[:, None]))

    residuals = frame["correct"].to_numpy(dtype=float) - fitted
    score_rows = design * residuals[:, None]
    clusters = frame["key"].astype(str).to_numpy()
    unique_clusters = np.unique(clusters)
    if unique_clusters.size < 2:
        raise ValueError("Clustered inference requires at least two distinct sample keys")

    meat = np.zeros((design.shape[1], design.shape[1]), dtype=float)
    for cluster in unique_clusters:
        cluster_score = score_rows[clusters == cluster].sum(axis=0)
        meat += np.outer(cluster_score, cluster_score)

    n = len(frame)
    n_samples = frame["key"].astype(str).nunique()
    n_parameters = design.shape[1]
    if n <= n_parameters:
        raise ValueError("Clustered inference requires more observations than fitted coefficients")
    correction = (n_samples / (n_samples - 1.0)) * ((n - 1.0) / (n - n_parameters))
    covariance = correction * (bread @ meat @ bread)
    se = float(np.sqrt(max(covariance[3, 3], 0.0)))
    beta_tv = float(beta[3])
    p_value = float(2.0 * norm.sf(abs(beta_tv / se))) if se > 0 else np.nan
    return InteractionFit(beta_tv, se, p_value, n, n_samples)


def classify_cdp(
    a_text: float,
    a_vision: float,
    a_joint: float,
    beta_tv: float,
    q_value: float,
    *,
    floor_threshold: float = 0.05,
    alpha: float = 0.05,
) -> str:
    """Apply the floor-first CDP assignment rule."""
    if min(a_text, a_vision) <= floor_threshold and a_joint <= floor_threshold:
        return "floor-limited"
    jas = min(a_text, a_vision) - a_joint
    if (
        a_joint < a_text
        and a_joint < a_vision
        and jas > 0
        and beta_tv < 0
        and q_value < alpha
    ):
        return "confirmed"
    return "unsupported"


def analyze_cdp(
    frame: pd.DataFrame,
    *,
    floor_threshold: float = 0.05,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Analyze every prespecified grid row in a long-form correctness table."""
    _validate(frame)
    records: list[dict[str, object]] = []
    for group_key, group in frame.groupby(GROUP_COLUMNS, sort=True, dropna=False):
        fit = fit_interaction(group)
        a_clean = _cell_accuracy(group, 0, 0)
        a_text = _cell_accuracy(group, 1, 0)
        a_vision = _cell_accuracy(group, 0, 1)
        a_joint = _cell_accuracy(group, 1, 1)
        records.append(
            {
                **dict(zip(GROUP_COLUMNS, group_key)),
                "n_samples": fit.n_samples,
                "A_clean": a_clean,
                "A_text": a_text,
                "A_vision": a_vision,
                "A_joint": a_joint,
                "JAS": min(a_text, a_vision) - a_joint,
                "beta_TV": fit.beta_tv,
                "se_TV": fit.se_tv,
                "p_TV": fit.p_tv,
            }
        )

    result = pd.DataFrame(records)
    result["q_TV"] = benjamini_hochberg(result["p_TV"])
    result["cdp_status"] = result.apply(
        lambda row: classify_cdp(
            float(row["A_text"]),
            float(row["A_vision"]),
            float(row["A_joint"]),
            float(row["beta_TV"]),
            float(row["q_TV"]),
            floor_threshold=floor_threshold,
            alpha=alpha,
        ),
        axis=1,
    )
    return result.sort_values(GROUP_COLUMNS).reset_index(drop=True)

