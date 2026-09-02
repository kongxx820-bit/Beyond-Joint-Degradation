import numpy as np
import pandas as pd

from bjd_cdp.diagnosis import benjamini_hochberg, classify_cdp, fit_interaction


def test_bh_is_monotone_in_ranked_order():
    p_values = np.array([0.04, 0.001, 0.02, np.nan])
    q_values = benjamini_hochberg(p_values)
    assert np.isnan(q_values[-1])
    ranked = q_values[np.argsort(p_values[:3])]
    assert np.all(np.diff(ranked) >= -1e-12)


def test_floor_rule_precedes_interaction_rule():
    status = classify_cdp(0.04, 0.70, 0.03, -3.0, 1e-8)
    assert status == "floor-limited"


def test_confirmed_requires_all_conditions():
    assert classify_cdp(0.30, 0.60, 0.10, -1.0, 0.01) == "confirmed"
    assert classify_cdp(0.30, 0.60, 0.10, +1.0, 0.01) == "unsupported"
    assert classify_cdp(0.30, 0.60, 0.35, -1.0, 0.01) == "unsupported"


def test_interaction_fit_uses_repeated_sample_clusters():
    rows = []
    for index in range(20):
        outcomes = {
            (0, 0): int(index != 0),
            (1, 0): int(index % 4 != 0),
            (0, 1): int(index % 5 != 0),
            (1, 1): int(index % 2 != 0),
        }
        for (text, vision), correct in outcomes.items():
            rows.append(
                {
                    "key": f"synthetic-{index}",
                    "correct": correct,
                    "T": text,
                    "V": vision,
                }
            )

    fit = fit_interaction(pd.DataFrame(rows))
    assert fit.n_observations == 80
    assert fit.n_samples == 20
    assert np.isfinite(fit.beta_tv)
    assert np.isfinite(fit.se_tv)
    assert fit.se_tv > 0

