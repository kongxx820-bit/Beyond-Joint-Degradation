# Beyond Joint Degradation

Official code-only reproducibility repository for **“Beyond Joint Degradation: Interaction-Aware Bilingual Diagnosis for Harmful Meme Detection.”**

**Paper:** published in **Data Intelligence** (2026), DOI [10.3724/2096-7004.di.2026.0616](https://doi.org/10.3724/2096-7004.di.2026.0616); accepted at **CCL 2026**, submission **DI-2026-0616**. See the [official accepted-paper list](https://cips-cl.org/static/CCL2026/en/cclProgram/accepted/index.html) and the [paper citation below](#paper-citation).

## Code-only release policy

This repository contains experiment code, configurations, unit tests, and detailed reproduction instructions. It intentionally contains **no experimental outputs**: no model predictions, per-sample correctness records, sample identifiers, metric files, result tables, generated figures, or paper-result summaries are tracked.

To reproduce an experiment, obtain the source datasets and model weights through their official channels, run the pipeline locally, and keep all generated files under ignored directories such as `data/private/`, `data/manifests/`, `data/pvd/`, `predictions/`, and `outputs/`.

## Included components

- `configs/classification_prompts.json`: bilingual binary-classification prompts and label mapping.
- `configs/pvd_parameters.yaml`: exact parameters for the PVD operators.
- `configs/tri_templates.json`: sanitized TRI examples and construction constraints.
- `src/bjd_cdp/pvd.py`: deterministic/seeded PVD implementation.
- `scripts/apply_pvd.py`: command-line PVD utility for one image.
- `scripts/build_condition_manifest.py`: construction of matched clean, text-only, vision-only, and joint conditions.
- `scripts/run_model_inference.py`: configurable Qwen2.5-VL, InternVL2.5, LLaVA-1.5, and OpenAI-compatible inference runner.
- `scripts/normalize_predictions.py`: strict parsing, four-condition matching, and local correctness-table construction.
- `src/bjd_cdp/diagnosis.py`: JAS, sample-clustered logistic interaction testing, BH-FDR correction, and CDP assignment.
- `scripts/run_cdp.py`: generation of CDP tables from locally generated prediction records.
- `tests/`: synthetic unit tests; these fixtures are not paper results.
- `docs/REPRODUCIBILITY_GUIDE_EN.md` and `docs/REPRODUCIBILITY_GUIDE_ZH.md`: detailed end-to-end instructions.
- `docs/DATA_CONSTRUCTION.md`: schemas, matching rules, perturbation construction, and prediction normalization.
- `docs/DATA_AND_LICENSING.md`: source-data access and licensing boundaries.
- `docs/REPRODUCIBILITY_SCOPE.md`: exact scope of the code-only release.

## Source datasets

- English: Hateful Memes, obtained through the official data channel.
- Chinese: ToxiCN-MM v2.0, obtained through its official repository and request process.

The source datasets are not redistributed here. Follow the links in [Data and licensing notes](docs/DATA_AND_LICENSING.md).

## Environment check

Python 3.10 or newer is supported. Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

The tests use only synthetic values and do not reveal or encode paper results.

## End-to-end reproduction outline

1. Obtain Hateful Memes and ToxiCN-MM under their respective licenses.
2. Prepare a private source manifest with the schema in `docs/DATA_CONSTRUCTION.md`.
3. Construct matched four-condition manifests and PVD images with `scripts/build_condition_manifest.py`.
4. Run each target model with `scripts/run_model_inference.py`, fixed prompts, and deterministic decoding.
5. Convert local predictions to the long-form CDP schema with `scripts/normalize_predictions.py`.
6. Run `scripts/run_cdp.py` to calculate JAS, the sample-clustered interaction test, BH-FDR-adjusted values, and CDP status.
7. Store all generated predictions, metrics, tables, and figures locally; do not commit them to this repository.

Start with the [English guide](docs/REPRODUCIBILITY_GUIDE_EN.md) or [Chinese guide](docs/REPRODUCIBILITY_GUIDE_ZH.md).

## Example commands

Build a matched condition manifest from licensed local data:

```bash
python scripts/build_condition_manifest.py \
  --input data/private/source_manifest.csv \
  --output data/manifests/conditions.csv \
  --pvd-dir data/pvd \
  --image-root data/private/images \
  --generate-pvd \
  --seed 2026
```

Run CDP after model inference and prediction normalization:

```bash
python scripts/run_cdp.py \
  --input predictions/aligned_correctness.csv \
  --output-dir outputs
```

The commands create local experimental outputs. The repository's `.gitignore` prevents the standard output locations from being tracked.

## Repository layout

```text
configs/                    prompts, TRI construction examples, and PVD parameters
docs/                       detailed reproduction and licensing instructions
scripts/                    construction, PVD, and CDP command-line programs
src/bjd_cdp/                reusable PVD and CDP implementation
tests/                      synthetic unit tests
```

## Citation and license

Please cite the paper and the software release you use. Machine-readable metadata is in [`CITATION.cff`](CITATION.cff): `preferred-citation` describes the paper, while the top-level metadata describes the software. GitHub's **Cite this repository** entry uses the preferred paper citation; a separate software citation is provided below.

### Paper citation

Xiangming Kong, Junyu Lu, Bo Xu, and Hongfei Lin. 2026. **Beyond Joint Degradation: Interaction-Aware Bilingual Diagnosis for Harmful Meme Detection.** *Data Intelligence*. DOI: [10.3724/2096-7004.di.2026.0616](https://doi.org/10.3724/2096-7004.di.2026.0616).

**Publication metadata verified on 2026-10-29:** the [Crossref DOI record](https://api.crossref.org/works/10.3724/2096-7004.di.2026.0616) matches the title and all four authors, identifies the work as a journal article in *Data Intelligence*, and records online publication on **2026-10-22**. The DOI resolves to the [publisher's article page](https://www.sciengine.com/doi/10.3724/2096-7004.di.2026.0616). The verified DOI record supplies no volume, issue, or page/article number, so those fields are omitted from the citation. CCL 2026 acceptance is independently documented under submission **DI-2026-0616** in the [official list](https://cips-cl.org/static/CCL2026/en/cclProgram/accepted/index.html).

```bibtex
@article{kong2026beyond,
  author = {Kong, Xiangming and Lu, Junyu and Xu, Bo and Lin, Hongfei},
  title = {{Beyond Joint Degradation: Interaction-Aware Bilingual Diagnosis for Harmful Meme Detection}},
  journal = {Data Intelligence},
  year = {2026},
  doi = {10.3724/2096-7004.di.2026.0616},
  url = {https://doi.org/10.3724/2096-7004.di.2026.0616}
}
```

### Software citation

The software version **1.0.0** and release date **2026-09-02** describe the code release. Cite that release separately when appropriate:

```bibtex
@software{kong2026beyondSoftware,
  author = {Kong, Xiangming and Lu, Junyu and Xu, Bo and Lin, Hongfei},
  title = {{Beyond Joint Degradation: Interaction-Aware Bilingual Diagnosis for Harmful Meme Detection}},
  version = {1.0.0},
  date = {2026-09-02},
  year = {2026},
  note = {Code-only reproducibility software; released 2026-09-02},
  url = {https://github.com/kongxx820-bit/Beyond-Joint-Degradation}
}
```

### License

Code is released under the MIT License. Dataset licenses remain separate and are not replaced by the software license.
