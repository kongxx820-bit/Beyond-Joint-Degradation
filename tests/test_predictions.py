import pandas as pd
import pytest

from bjd_cdp.predictions import normalize_prediction_frames, parse_binary_response


def _synthetic_predictions() -> pd.DataFrame:
    rows = []
    for key, label in (("sample-a", 0), ("sample-b", 1)):
        for text, vision in ((0, 0), (1, 0), (0, 1), (1, 1)):
            rows.append(
                {
                    "key": key,
                    "label": label,
                    "pred": label if (text, vision) != (1, 1) else 1 - label,
                    "T": text,
                    "V": vision,
                    "model": "synthetic-model",
                    "language": "synthetic-language",
                    "pvd": "synthetic-pvd",
                    "tri": "synthetic-tri",
                }
            )
    return pd.DataFrame(rows)


def test_binary_parser_is_strict():
    assert parse_binary_response("harmful") == 1
    assert parse_binary_response("non-harmful") == 0
    assert parse_binary_response("**harmful**") == 1
    assert parse_binary_response("The answer is harmful") is None


def test_normalizer_builds_all_four_conditions():
    normalized, exclusions = normalize_prediction_frames([_synthetic_predictions()])
    assert exclusions.empty
    assert len(normalized) == 8
    assert set(normalized["condition"]) == {"clean", "text", "vision", "joint"}
    assert normalized.groupby("key").size().eq(4).all()


def test_normalizer_rejects_incomplete_samples_by_default():
    incomplete = _synthetic_predictions().iloc[:-1].copy()
    with pytest.raises(ValueError, match="incomplete matched samples"):
        normalize_prediction_frames([incomplete])
