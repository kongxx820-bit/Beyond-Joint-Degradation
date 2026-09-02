# Reproducibility and Usage Guide

## 1. Release boundary

This is a code-only release. It includes the implementation and the information needed to rerun the experimental pipeline, but it does not include any paper output or intermediate experimental record. In particular, the repository contains no source samples, sample identifiers, perturbed samples, model predictions, correctness rows, metrics, tables, figures, checkpoints, or reported-result summaries.

Reproduction therefore requires legitimate access to the source datasets and to the selected vision-language models. The generated artifacts must remain in local directories ignored by Git.

## 2. Pipeline overview

The experimental pipeline has six stages:

1. obtain and validate the licensed source datasets;
2. prepare one private source manifest;
3. construct the matched clean, text-only, vision-only, and joint conditions;
4. run the models with fixed prompts and deterministic decoding;
5. normalize model responses into a paired long-form correctness table; and
6. calculate JAS, sample-clustered logistic interaction evidence, BH-FDR-adjusted values, and CDP status.

The repository implements stages 3--6 directly. Model weights, commercial API access, and licensed datasets are external dependencies and are not bundled.

## 3. Software environment

Recommended base environment:

- Python 3.11;
- NumPy, pandas, and SciPy for CDP analysis;
- OpenCV for PVD generation;
- model-specific libraries for inference.

Create the analysis environment:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

Install PVD support:

```bash
python -m pip install -e ".[pvd]"
```

For model inference, create separate environments if package requirements conflict. Record the following for every run:

- operating system and Python version;
- GPU model, CUDA version, and driver version;
- `torch`, `transformers`, and model-specific package versions;
- model repository and immutable revision/commit;
- precision and device mapping;
- image preprocessing parameters;
- decoding parameters;
- random seed;
- prompt-file checksum.

Do not place API credentials in code or configuration files. Use environment variables provided by the relevant model service.

## 4. Obtain the source data

Obtain Hateful Memes and ToxiCN-MM through their official channels. Do not copy their raw images or text into this repository. Place locally licensed material under `data/private/`, which is ignored by Git.

Recommended private layout:

```text
data/private/
  hateful_memes/
    images/
    source.csv
  toxicn_mm/
    images/
    source.csv
```

See `docs/DATA_AND_LICENSING.md` before using or sharing either dataset.

## 5. Prepare the private source manifest

Create a UTF-8 CSV containing exactly one row per source sample. Required columns are:

| Column | Meaning |
|---|---|
| `sample_id` | Stable identifier unique within the manifest |
| `label` | Binary gold label: `1` harmful, `0` non-harmful |
| `external_text` | Original external text supplied with the image |
| `image_path` | Absolute path or path relative to `--image-root` |
| `language` | `english` or `chinese` |
| `tri_neutral_text` | Prespecified neutral-context replacement text |
| `tri_positive_text` | Prespecified risk-oriented replacement text |

TRI replaces the external-text channel. It must not be appended to the original text. Keep the gold label fixed across all matched conditions. Do not include the answer label or a direct synonym of it in a TRI template.

Additional construction details and validation rules are in `docs/DATA_CONSTRUCTION.md`.

## 6. Generate matched conditions and PVD images

Run:

```bash
python scripts/build_condition_manifest.py \
  --input data/private/source_manifest.csv \
  --output data/manifests/conditions.csv \
  --pvd-dir data/pvd \
  --image-root data/private/images \
  --operators sp_text sp_image spread \
  --seed 2026 \
  --generate-pvd
```

For every source sample, PVD family, and TRI family, the script creates four matched records:

| Condition | `T` | `V` | Image | External text |
|---|---:|---:|---|---|
| clean | 0 | 0 | original | original |
| text | 1 | 0 | original | TRI replacement |
| vision | 0 | 1 | PVD image | original |
| joint | 1 | 1 | PVD image | TRI replacement |

PVD parameters are stored in `configs/pvd_parameters.yaml`. Stochastic operators receive a stable per-sample seed derived from the base seed, sample ID, and operator name, so input-row order does not change the generated image.

To inspect one PVD transformation:

```bash
python scripts/apply_pvd.py \
  --input data/private/images/example.jpg \
  --output data/pvd/example_spread.jpg \
  --operator spread \
  --seed 2026
```

## 7. Run model inference

Use `configs/classification_prompts.json` for both language tracks. Each manifest row must be evaluated independently with the row's `image_path` and `external_text`.

Required inference policy:

- use the same model revision for all four conditions in a matched group;
- use the same preprocessing and decoding settings across conditions;
- use deterministic decoding (`do_sample=false` or temperature zero, as supported);
- request exactly one of `harmful` and `non-harmful`;
- preserve the unmodified raw response for local auditing;
- record parser failures separately instead of converting them silently to a class;
- resume by stable row key and never overwrite a completed row without recording the rerun.

For open-weight models, pin the Hugging Face revision. For API models, record the provider, model identifier, date, region if applicable, and request parameters. API nondeterminism and model updates can prevent bitwise-identical responses even under temperature-zero decoding.

Store inference output under `predictions/`. This directory is ignored by Git.

Install model-runner dependencies after installing a CUDA-compatible PyTorch build for the machine:

```bash
python -m pip install -r requirements-inference.txt
```

Example Qwen2.5-VL run:

```bash
python scripts/run_model_inference.py \
  --input data/manifests/conditions.csv \
  --output predictions/qwen25vl.csv \
  --backend qwen25vl \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --model-label Qwen2.5-VL-3B \
  --revision <PINNED_COMMIT> \
  --image-root data/private
```

Use `--backend internvl25` for InternVL2.5, `--backend llava15` for LLaVA-1.5, or `--backend openai-compatible` for an OpenAI-compatible multimodal endpoint. For an API run, supply the key through `OPENAI_API_KEY` or change the variable name with `--api-key-env`. Never commit the generated CSV.

## 8. Normalize predictions

The CDP analysis input is a UTF-8 CSV with the following required columns:

| Column | Meaning |
|---|---|
| `key` | Stable sample key shared by all four conditions |
| `correct` | `1` if the parsed prediction equals the gold label, otherwise `0` |
| `T` | Text perturbation indicator |
| `V` | Visual perturbation indicator |
| `model` | Model identifier |
| `language` | Language track |
| `pvd` | PVD family |
| `tri` | TRI family |

Recommended audit columns are `label`, `pred`, `condition`, `raw_output`, `parser_status`, `model_revision`, `seed`, and `prompt_sha256`. The analysis ignores extra columns.

Before analysis, verify that each `(model, language, pvd, tri, key)` group contains one and only one record for each `(T,V)` cell: `(0,0)`, `(1,0)`, `(0,1)`, and `(1,1)`. Construct the four-way intersection first; do not calculate different conditions on different sample sets.

Normalize one or more local inference files:

```bash
python scripts/normalize_predictions.py \
  --input predictions/qwen25vl.csv predictions/internvl25.csv \
  --output predictions/aligned_correctness.csv
```

The default policy stops on missing or duplicate cells. If dropping incomplete matched samples was specified before analysis, use `--incomplete-policy drop` together with `--exclusion-log predictions/exclusions.csv`.

## 9. Run CDP locally

```bash
python scripts/run_cdp.py \
  --input predictions/aligned_correctness.csv \
  --output-dir outputs \
  --floor 0.05 \
  --alpha 0.05
```

The command writes locally generated analysis tables under `outputs/`. These files are ignored by Git and are not part of the public repository.

For each analysis row, the implementation calculates:

```text
JAS = min(A_text, A_vision) - A_joint
```

It then fits the binary interaction model

```text
logit P(correct = 1) = beta_0 + beta_T T + beta_V V + beta_TV T*V
```

and calculates a sample-clustered sandwich covariance matrix using `key` as the cluster identifier. This accounts for the four repeated conditions belonging to the same source sample. Interaction p-values are adjusted with Benjamini-Hochberg correction over the supplied analysis grid.

The floor-first CDP rule is:

1. assign `floor-limited` when the prespecified floor condition is met;
2. otherwise assign `confirmed` only when the joint cell is worse than both matched unimodal cells, JAS is positive, the interaction coefficient is negative, and its corrected value is below `alpha`;
3. assign all remaining non-floor rows to `unsupported`.

Do not change the floor threshold, FDR family, parser rule, or sample intersection after inspecting the results. Sensitivity checks must be labeled separately from the prespecified analysis.

## 10. Quality-control checklist

Before accepting a local reproduction run, verify:

- source IDs are unique before condition expansion;
- labels remain unchanged across conditions;
- PVD images are readable and have the expected dimensions and channels;
- TRI is replacement-only and contains no answer-bearing label term;
- each matched key has all four conditions;
- no duplicate `(key,T,V)` exists within a grid row;
- model revision and prompt checksum are constant within a comparison;
- parsing accepts only the declared binary labels;
- parser failures and missing images are reported;
- FDR is calculated over the intended analysis family;
- all generated outputs remain untracked by Git.

Useful checks:

```bash
git status --short
python -m pytest -q
python scripts/run_cdp.py --help
python scripts/build_condition_manifest.py --help
```

## 11. Troubleshooting

### Missing required columns

Compare the input with the schema in Section 8. Column names are case-sensitive.

### Duplicate sample-condition rows

Resolve duplicates with a documented stable rule before analysis. Never keep an arbitrary row based on filesystem order.

### Missing four-condition cells

Take the exact four-way key intersection within each analysis row. Report exclusions locally and do not impute predictions.

### Complete separation

A cell with all-correct or all-incorrect outcomes can make ordinary finite logistic estimates unavailable. Preserve the row, record the failure, and apply only a correction method that was specified before result inspection.

### Non-identical rerun predictions

Check model revision, CUDA and package versions, image preprocessing, precision, prompt bytes, decoding settings, API model version, and parser rules. Statistical reproduction and bitwise-identical model generation are different reproducibility targets.

## 12. Sharing a local reproduction

If another researcher needs your generated outputs, share them outside this GitHub repository under the source-dataset licenses and your institution's data-governance rules. Remove raw text, image paths, credentials, and unrestricted identifiers before any transfer. This public repository remains code-only.
