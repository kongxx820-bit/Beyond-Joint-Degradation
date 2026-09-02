# Data Construction Protocol

## 1. Scope

This document describes how to reconstruct the matched clean, text-only, vision-only, and joint experimental conditions after obtaining the source datasets through their official channels. This code-only repository does not redistribute Hateful Memes or ToxiCN-MM images, text, identifiers, predictions, correctness records, or experimental results.

The public repository provides deterministic PVD code, exact PVD parameters, classification prompts, sanitized TRI construction examples, a batch condition-manifest builder, and statistical analysis scripts. Researchers remain responsible for source-dataset access, TRI selection, model access, and inference.

## 2. Source datasets and exact split

The reported bilingual grid uses the official training split of each source dataset as its source pool. Raw images and meme text are not part of this repository.

| Track | Source dataset | Version and split used | Source-pool size | Binary label mapping | Official access |
|---|---|---|---:|---|---|
| English | Hateful Memes Challenge | official training split (`train.jsonl`, locally normalized to `train.csv`) | 8,500 | `1` (hateful) -> harmful; `0` (not hateful) -> non-harmful | [Meta dataset description](https://ai.meta.com/blog/hateful-memes-challenge-and-data-set/) |
| Chinese | ToxiCN-MM | version 2.0 training split (`train_data_discription_2.0.json`) | 9,600 | `1` (harmful) -> harmful; `0` (non-harmful) -> non-harmful | [official ToxiCN-MM repository and request form](https://github.com/DUT-lujunyu/ToxiCN_MM) |

For both tracks, the source record is normalized to a stable sample key, language, image path, external text, and binary reference label. No new annotation is introduced by this release.

Apply the paper's prespecified sampling rule locally and save the selected identifiers in a private manifest. Exact source keys are intentionally not published. If any condition is missing after inference, take the strict four-condition intersection within that grid row and record the exclusion locally; do not impute a prediction.

## 3. Reported main-grid recipe

The 24 reported settings are the Cartesian product of:

- two models: `Qwen2.5-VL-3B` and `InternVL2.5-4B`;
- two language tracks: English and Chinese;
- three main PVD operators: `sp_text`, `sp_image`, and `spread`; and
- two prespecified TRI families: `neutral_text` and `positive_text`.

Each setting contains the four matched clean, text-only, vision-only, and joint conditions described below. Inference uses deterministic decoding and the language-specific binary classification prompt in `configs/classification_prompts.json`. The same model version, prompt, decoding configuration, parser, and source keys must be held fixed across the four conditions.

## 4. Final PVD construction used by the reported experiments

The public implementation in `src/bjd_cdp/pvd.py` and the machine-readable parameters in `configs/pvd_parameters.yaml` correspond to the final bilingual `attacked_train` generator used by the reported Qwen and InternVL evaluations.

| Operator | Image transformation |
|---|---|
| `sp_text` | Process the top and bottom 26% of the image with salt-and-pepper noise (`amount=0.08`), Gaussian noise (`sigma=28`), Gaussian blur (`7x7`, `sigma=2.5`), and contrast (`alpha=0.68`, `beta=6`); then apply mild global contrast (`alpha=0.93`, `beta=2`). |
| `sp_image` | Apply full-image Gaussian noise (`sigma=34`), salt-and-pepper noise (`amount=0.07`), and contrast (`alpha=0.82`, `beta=4`). |
| `spread` | Downsample to 50% with area interpolation, upsample with linear interpolation, apply Gaussian blur (`15x15`, `sigma=5.5`), a horizontal motion-blur kernel of 15, and contrast (`alpha=0.78`, `beta=5`). |
| `newsprint` | Auxiliary-only operator: grayscale, downsample to 58%, Gaussian noise (`sigma=14`), salt-and-pepper noise (`amount=0.025`), Gaussian blur (`5x5`, `sigma=1.2`), and contrast (`alpha=0.72`, `beta=10`). It is not one of the three main 24-grid PVD families. |

All pixel operations use uint8 BGR images. The release derives a deterministic seed from the base seed, source key, and operator, so the generated PVD image does not depend on manifest row order.

## 5. Four matched conditions

For each source sample, PVD operator, and TRI family, construct four cells while keeping the sample label fixed:

| Condition | `T` | `V` | External text | Image |
|---|---:|---:|---|---|
| `clean` | 0 | 0 | original external text | original image |
| `text` | 1 | 0 | prespecified TRI replacement | original image |
| `vision` | 0 | 1 | original external text | PVD image |
| `joint` | 1 | 1 | prespecified TRI replacement | PVD image |

TRI replaces the external-text channel. It is not appended to the original text. The same sample key, reference label, model version, classification prompt, decoding configuration, and output parser must be used across all four cells.

## 6. Licensed local source manifest

Create a private CSV with one row per source sample and the following fields:

| Field | Requirement |
|---|---|
| `sample_id` | Stable, unique, non-empty identifier |
| `label` | Binary reference label: 0 or 1 |
| `external_text` | Original external text supplied to the model |
| `image_path` | Absolute path or path relative to the source CSV |
| `language` | `english` or `chinese` |
| `tri_neutral_text` | Prespecified label-free neutral TRI replacement |
| `tri_positive_text` | Prespecified label-free risk-oriented TRI replacement |

Example schema only:

```csv
sample_id,label,external_text,image_path,language,tri_neutral_text,tri_positive_text
sample_0001,1,<licensed text>,images/sample_0001.jpg,english,<neutral TRI>,<risk-oriented TRI>
```

Do not commit this manifest. It contains licensed source content and local paths. The repository ignores `data/private/`, `data/manifests/`, and `data/pvd/` by default.

## 7. Build the four-condition manifest

Create only the manifest paths first:

```bash
python scripts/build_condition_manifest.py \
  --input data/private/source_manifest.csv \
  --output data/manifests/four_condition_manifest.csv \
  --pvd-dir data/pvd \
  --seed 2026
```

This dry run writes the matched rows without creating image files. For `N` source samples, three PVD operators, two TRI families, and four conditions, the output contains `N × 3 × 2 × 4` rows.

To generate the PVD images as well:

```bash
python -m pip install -e '.[pvd]'
python scripts/build_condition_manifest.py \
  --input data/private/source_manifest.csv \
  --output data/manifests/four_condition_manifest.csv \
  --pvd-dir data/pvd \
  --seed 2026 \
  --generate-pvd
```

On Windows PowerShell, use double quotes around the optional dependency:

```powershell
python -m pip install -e ".[pvd]"
python scripts/build_condition_manifest.py --input data/private/source_manifest.csv --output data/manifests/four_condition_manifest.csv --pvd-dir data/pvd --seed 2026 --generate-pvd
```

The supported main-grid operators are `sp_text`, `sp_image`, and `spread`. Parameters are recorded in `configs/pvd_parameters.yaml`. The script derives a deterministic PVD seed from the base seed, sample key, and operator, so the generated image is independent of input-row order.

## 8. Output manifest

The builder writes:

| Field | Meaning |
|---|---|
| `key` | Stable source sample identifier |
| `label` | Fixed binary reference label |
| `language` | English or Chinese track |
| `pvd` | PVD operator family |
| `tri` | TRI family |
| `condition` | clean, text, vision, or joint |
| `T`, `V` | Factorial design indicators |
| `external_text` | Text actually supplied in this condition |
| `image_path` | Image actually supplied in this condition |
| `source_image_path` | Original licensed image reference |
| `pvd_seed` | Deterministic per-sample/operator seed |

The construction manifest is private working data because it includes source text and paths. It must not be committed to the public repository.

## 9. TRI requirements

`configs/tri_templates.json` contains sanitized examples and construction principles, not a claim that one template represents a real platform distribution. Final TRI strings must be determined before examining significance and recorded in the private source manifest.

For each TRI, record its language, pragmatic direction, intensity level, template family, text length, and stable template identifier. A TRI must remain label-free and must not directly state `harmful` or `non-harmful`.

The released builder deliberately does not synthesize TRI text. Automatic synthesis during reconstruction would introduce an uncontrolled model, prompt, and sampling dependency.

## 10. Model inference

Use the corresponding prompt in `configs/classification_prompts.json`. Across the four matched conditions, hold constant:

- model name and exact weight or API revision;
- image preprocessing version;
- prompt;
- decoding parameters;
- random seed, if applicable;
- output parser; and
- retry/failure policy.

Save raw responses in private logs, then strictly parse the public analysis value as `harmful` (1) or `non-harmful` (0). Parsing failures must be recorded explicitly and must not be silently mapped to either class.

The configurable runner `scripts/run_model_inference.py` supports Qwen2.5-VL, InternVL2.5, LLaVA-1.5, and OpenAI-compatible multimodal endpoints. Pin the model revision with `--revision`, keep outputs under `predictions/`, and use `scripts/normalize_predictions.py` to enforce the four-condition schema before CDP analysis.

## 11. Construct the local analysis table

After inference:

1. normalize sample keys without retaining local paths;
2. deduplicate each `key × model × language × pvd × tri × condition` cell using a documented stable rule;
3. take the four-condition intersection within each grid row;
4. compute `correct = int(label == pred)`;
5. retain only `key, label, pred, correct, condition, T, V, model, language, pvd, tri`; and
6. exclude raw text, image paths, image bytes, raw model responses, credentials, and private annotation records.

Save this table under `predictions/` or another ignored local directory. Do not commit it because it is an experimental output and may retain restricted identifiers.

## 12. Statistical analysis

Run:

```bash
python scripts/run_cdp.py \
  --input predictions/aligned_correctness.csv \
  --output-dir outputs
```

The implementation computes JAS, fits the logistic interaction model, calculates sample-clustered standard errors using `key`, applies BH-FDR over the supplied grid, and assigns CDP status. Generated tables remain local under the ignored `outputs/` directory.

## 13. Reproducibility checklist

- Source datasets obtained through official channels.
- Hateful Memes training split (8,500) and ToxiCN-MM v2.0 training split (9,600) recorded.
- Prespecified local sampling manifest applied separately to each language.
- Sample keys unique before condition expansion.
- TRI texts frozen and logged before statistical testing.
- PVD parameters and base seed recorded.
- Four matched conditions generated for each retained sample.
- Model, prompt, decoding, and parser fixed across conditions.
- Parse failures and deduplication decisions logged.
- Four-condition intersection computed within each grid row.
- Generated artifacts stripped of credentials before any authorized transfer.
- CDP tables generated locally and left untracked by Git.
