# Reproducibility scope

## What this repository provides

This repository provides a code-only implementation of the experimental protocol:

- deterministic and seeded PVD operators;
- machine-readable PVD parameters;
- bilingual classification prompts and label mapping;
- sanitized TRI construction principles;
- matched four-condition manifest construction;
- configurable inference runners for the reported model families;
- strict response parsing and four-condition prediction normalization;
- JAS calculation;
- sample-clustered logistic interaction testing;
- BH-FDR correction and the floor-first CDP rule;
- table-generation code; and
- synthetic unit tests plus detailed English and Chinese instructions.

## What this repository does not provide

No experimental result or research data is committed. Excluded content includes:

- raw or transformed images and text;
- source sample identifiers or sampling manifests;
- model predictions and free-form responses;
- correctness tables;
- metrics, coefficients, p-values, q-values, or status tables;
- paper figures or generated result tables;
- model weights and checkpoints; and
- API credentials or private annotation records.

The standard local directories for these materials are listed in `.gitignore`.

## Reproduction levels

### Protocol and implementation verification

Anyone can install the package and run the synthetic unit tests:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

This verifies schema validation, stable condition construction, PVD seeding, BH-FDR behavior, floor-first assignment, and sample-clustered interaction fitting without exposing paper results.

### End-to-end experimental reproduction

End-to-end reproduction additionally requires:

1. separately licensed Hateful Memes and ToxiCN-MM data;
2. the prespecified local sampling manifest;
3. the target model weights or authorized API access;
4. pinned model and software revisions;
5. local model inference for every matched condition; and
6. local CDP analysis of the generated predictions.

The complete procedure is documented in `REPRODUCIBILITY_GUIDE_EN.md`, `REPRODUCIBILITY_GUIDE_ZH.md`, and `DATA_CONSTRUCTION.md`.

## Reproducibility limitations

Bitwise-identical model responses cannot be guaranteed when external weights, inference kernels, drivers, provider-side API models, or dataset releases change. A rigorous rerun should record model commits, package locks, CUDA and driver versions, image preprocessing, prompts, decoding settings, parser behavior, missing files, retries, and exclusions.

This repository supports transparent reconstruction of the protocol. It does not claim that code alone can recreate experimental outputs without the external data and models used by the researcher.
