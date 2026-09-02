from pathlib import Path

import pandas as pd

from scripts.build_condition_manifest import build_manifest


def _source(sample_ids=("sample-a",)) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "label": index % 2,
                "external_text": f"original-{sample_id}",
                "image_path": f"images/{sample_id}.png",
                "language": "english",
                "tri_neutral_text": f"neutral-{sample_id}",
                "tri_positive_text": f"risk-{sample_id}",
            }
            for index, sample_id in enumerate(sample_ids)
        ]
    )


def test_builder_expands_one_sample_to_the_24_cell_grid(tmp_path: Path):
    manifest = build_manifest(
        source=_source(),
        operators=("sp_text", "sp_image", "spread"),
        pvd_dir=tmp_path / "pvd",
        image_root=tmp_path,
        base_seed=2026,
        generate_pvd=False,
    )

    assert len(manifest) == 24
    assert manifest.groupby(["pvd", "tri"]).size().eq(4).all()
    assert set(manifest["condition"]) == {"clean", "text", "vision", "joint"}
    assert (manifest.loc[manifest["T"].eq(0), "external_text"] == "original-sample-a").all()
    assert (manifest.loc[manifest["T"].eq(1) & manifest["tri"].eq("neutral_text"), "external_text"] == "neutral-sample-a").all()
    assert manifest.loc[manifest["V"].eq(1), "image_path"].str.contains("pvd").all()


def test_pvd_seed_is_independent_of_input_order(tmp_path: Path):
    first = _source(("sample-a", "sample-b"))
    second = first.iloc[::-1].reset_index(drop=True)

    kwargs = {
        "operators": ("sp_text", "sp_image", "spread"),
        "pvd_dir": tmp_path / "pvd",
        "image_root": tmp_path,
        "base_seed": 2026,
        "generate_pvd": False,
    }
    left = build_manifest(first, **kwargs)[["key", "pvd", "pvd_seed"]].drop_duplicates()
    right = build_manifest(second, **kwargs)[["key", "pvd", "pvd_seed"]].drop_duplicates()

    pd.testing.assert_frame_equal(
        left.sort_values(["key", "pvd"]).reset_index(drop=True),
        right.sort_values(["key", "pvd"]).reset_index(drop=True),
    )
