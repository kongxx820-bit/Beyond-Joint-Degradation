#!/usr/bin/env python3
"""Apply one PVD operator to a single image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from bjd_cdp.pvd import OPERATORS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operator", choices=sorted(OPERATORS), required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    image = cv2.imread(str(args.input))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")
    output = OPERATORS[args.operator](image, np.random.default_rng(args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), output):
        raise OSError(f"Could not write image: {args.output}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()

