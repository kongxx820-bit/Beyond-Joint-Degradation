#!/usr/bin/env python3
"""Build a matched clean/text/vision/joint manifest from licensed local data.

The script does not download raw datasets or invent TRI text. The caller must
provide one original external-text field and the two prespecified TRI fields for
every sample. PVD images can be generated with the released operators or only
referenced in a dry-run manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "sample_id",
    "label",
    "external_text",
    "image_path",
    "language",
    "tri_neutral_text",
    "tri_positive_text",
}
TRI_COLUMNS = {
    "neutral_text": "tri_neutral_text",
    "positive_text": "tri_positive_text",
}
CONDITIONS = (
    ("clean", 0, 0),
    ("text", 1, 0),
    ("vision", 0, 1),
    ("joint", 1, 1),
)
DEFAULT_OPERATORS = ("sp_text", "sp_image", "spread")


def _stable_seed(base_seed: int, sample_id: str, operator: str) -> int:
    payload = f"{base_seed}\0{sample_id}\0{operator}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _safe_image_name(sample_id: str, source_suffix: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("._")[:72]
    if not stem:
        stem = "sample"
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:10]
    suffix = source_suffix.lower() if source_suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} else ".png"
    return f"{stem}-{digest}{suffix}"


def _validate_source(source: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(source.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if source.empty:
        raise ValueError("The source manifest is empty")

    source = source.copy()
    source["sample_id"] = source["sample_id"].astype("string").str.strip()
    if source["sample_id"].isna().any() or (source["sample_id"] == "").any():
        raise ValueError("sample_id must be non-empty")
    duplicates = source.loc[source["sample_id"].duplicated(keep=False), "sample_id"].unique()
    if len(duplicates):
        preview = ", ".join(map(str, duplicates[:5]))
        raise ValueError(f"sample_id must be unique; duplicate examples: {preview}")

    numeric_label = pd.to_numeric(source["label"], errors="coerce")
    if numeric_label.isna().any() or not numeric_label.isin([0, 1]).all():
        raise ValueError("label must contain only 0 or 1")
    source["label"] = numeric_label.astype(int)

    source["language"] = source["language"].astype("string").str.lower().str.strip()
    if not source["language"].isin(["english", "chinese"]).all():
        raise ValueError("language must be 'english' or 'chinese'")

    text_columns = ["external_text", "image_path", *TRI_COLUMNS.values()]
    for column in text_columns:
        source[column] = source[column].astype("string").str.strip()
        if source[column].isna().any() or (source[column] == "").any():
            raise ValueError(f"{column} must be non-empty for every sample")
    return source


def _generate_pvd_image(
    source_path: Path,
    output_path: Path,
    operator: str,
    seed: int,
) -> None:
    try:
        import cv2
        from bjd_cdp.pvd import OPERATORS
    except ImportError as exc:
        raise RuntimeError(
            "PVD generation requires the optional image dependency. "
            "Install it with: python -m pip install -e \".[pvd]\""
        ) from exc

    image = cv2.imread(str(source_path))
    if image is None:
        raise FileNotFoundError(f"Could not read source image: {source_path}")
    result = OPERATORS[operator](image, np.random.default_rng(seed))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        raise OSError(f"Could not write PVD image: {output_path}")


def build_manifest(
    source: pd.DataFrame,
    operators: tuple[str, ...],
    pvd_dir: Path,
    image_root: Path,
    base_seed: int,
    generate_pvd: bool,
) -> pd.DataFrame:
    source = _validate_source(source)
    records: list[dict[str, object]] = []

    for row in source.itertuples(index=False):
        sample_id = str(row.sample_id)
        source_image_text = str(row.image_path)
        source_image = Path(source_image_text)
        resolved_source = source_image if source_image.is_absolute() else image_root / source_image

        for operator in operators:
            seed = _stable_seed(base_seed, sample_id, operator)
            pvd_path = pvd_dir / operator / _safe_image_name(sample_id, source_image.suffix)
            if generate_pvd:
                _generate_pvd_image(resolved_source, pvd_path, operator, seed)

            for tri_name, tri_column in TRI_COLUMNS.items():
                tri_text = str(getattr(row, tri_column))
                for condition, text_applied, vision_applied in CONDITIONS:
                    records.append(
                        {
                            "key": sample_id,
                            "label": int(row.label),
                            "language": str(row.language),
                            "pvd": operator,
                            "tri": tri_name,
                            "condition": condition,
                            "T": text_applied,
                            "V": vision_applied,
                            "external_text": tri_text if text_applied else str(row.external_text),
                            "image_path": str(pvd_path) if vision_applied else source_image_text,
                            "source_image_path": source_image_text,
                            "pvd_seed": seed,
                        }
                    )

    manifest = pd.DataFrame.from_records(records)
    expected_rows = len(source) * len(operators) * len(TRI_COLUMNS) * len(CONDITIONS)
    if len(manifest) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, produced {len(manifest)}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Licensed source manifest CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output four-condition CSV")
    parser.add_argument(
        "--pvd-dir",
        type=Path,
        required=True,
        help="Directory for generated or referenced PVD images",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Base directory for relative image_path values; defaults to the input CSV directory",
    )
    parser.add_argument(
        "--operators",
        nargs="+",
        choices=DEFAULT_OPERATORS,
        default=list(DEFAULT_OPERATORS),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--generate-pvd",
        action="store_true",
        help="Generate PVD image files; without this flag only the manifest paths are written",
    )
    args = parser.parse_args()

    source = pd.read_csv(args.input, dtype={"sample_id": "string"})
    image_root = args.image_root if args.image_root is not None else args.input.parent
    manifest = build_manifest(
        source=source,
        operators=tuple(args.operators),
        pvd_dir=args.pvd_dir,
        image_root=image_root,
        base_seed=args.seed,
        generate_pvd=args.generate_pvd,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False, encoding="utf-8")

    print(f"Wrote {len(manifest)} condition rows to {args.output.resolve()}")
    print(f"Samples: {manifest['key'].nunique()}")
    print(f"Operators: {', '.join(args.operators)}")
    print("PVD images: generated" if args.generate_pvd else "PVD images: paths only (dry run)")


if __name__ == "__main__":
    main()

