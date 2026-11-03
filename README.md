# Gaia DR3 Catalogue Spectral-Type Classification with Ensemble Learning

**Authors:** Alberto Gudiño-Ochoa and Raquel Ochoa-Ornelas  
**Contact:** [alberto.gudino9401@alumnos.udg.mx](mailto:alberto.gudino9401@alumnos.udg.mx) · [raquel.oo@cdguzman.tecnm.mx](mailto:raquel.oo@cdguzman.tecnm.mx)

## Overview

Reproducible Python workflow for classifying seven Gaia DR3 catalogue-derived spectral labels (**O, B, A, F, G, K, M**). The repository includes the complete quality-filtering and feature-engineering code, a comparison of **ten classifiers** across three feature sets, three ensemble strategies, a feature–label circularity ablation, class-subsampling and weighting sensitivity experiments, and SHAP interpretability. An **optional** notebook compares predictions with matched LAMOST DR11 spectroscopic classifications.

**Scientific scope:** internal results assess agreement with Gaia-derived `SpType-ELS` catalogue labels. They are **not** independently verified spectroscopic classification accuracy. The external comparison is conditional on availability and quality of the matched LAMOST sample.

## Dataset: Kaggle

The data are obtained **separately from Kaggle** and are not redistributed here.

- **Kaggle dataset:** [Gaia 626k stars from DR3 (Data release 3) V2](https://www.kaggle.com/datasets/realkiller69/gaia-stars-dataset-from-dr3-data-release-3), as cited in the revised manuscript.
- **Expected input CSV filename:** `dataGaia2.csv` in the authors' original notebook (rename the Kaggle CSV or change `data_file` in `config/experiment.json` if its downloaded name differs).
- **Input location:** `data/dataGaia2.csv`.
- **Expected original cohort (authors' run):** 626,016 sources; 524,187 after quality filtering; 504,306 after requiring the three retained FLAME attributes; 35,000 after balancing to 5,000 per class.
- **Mandatory fields:** `Source`, `SpType-ELS`, `Plx`, `Dist`, `RUWE`, `Gmag`, `BPmag`, `RPmag`, `AG`, `ABP`, `ARP`, `Pstar`, `PGal`, `Teff`, `logg`, `[Fe/H]`, `Rad`, `EWHa`, `Rad-Flame`, `Lum-Flame`, `Mass-Flame`, `RA_ICRS`, `DE_ICRS`.

Optionally download the dataset using the Kaggle CLI (with Kaggle credentials configured):

```bash
python -m pip install kaggle
kaggle datasets download -d realkiller69/gaia-stars-dataset-from-dr3-data-release-3 -p data --unzip
```

After downloading, ensure the source CSV exists at `data/dataGaia2.csv` (or update `data_file`). Cite the original Kaggle dataset and any underlying Gaia sources according to their stated terms. Retain Gaia `Source` identifiers as **64-bit integers**, not floating-point values.

## Environment and installation

The source notebook records **Python 3.11.14** on macOS Apple Silicon with **NumPy 1.26.4, pandas 2.3.3, scikit-learn 1.4.2, CatBoost 1.2.8, LightGBM 4.6.0, XGBoost 3.1.3, TensorFlow 2.16.2, SHAP 0.51.0, and Astropy 7.2.0**. The pinned dependencies and remaining supported ranges are in `requirements.txt`.

```bash
# Run these commands from the repository root.
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name gaia-dr3-repro --display-name "Python 3.11 (Gaia DR3)"
jupyter notebook
```

On Windows activate the environment using `.venv\Scripts\activate` instead. Select the new notebook kernel if prompted. Neural-network training may require platform-specific TensorFlow setup; the authors' source environment used Apple Silicon.

## Execution order

Run the notebooks in the order below, using **Kernel → Restart & Run All** for each notebook. Do not skip earlier notebooks unless their `outputs/` files already exist.

| Notebook | Purpose | Main outputs |
|---|---|---|
| `notebooks/01_Data_Preprocessing.ipynb` | Input validation, Gaia quality cuts, observed and extinction-corrected feature engineering, FLAME-complete and balanced cohorts | `preprocessed.joblib`, `class_distribution.csv` |
| `notebooks/02_Exploratory_Analysis.ipynb` | H–R and sky-density diagrams, colour–colour features, correlations, optional t-SNE | `figures/*.png` |
| `notebooks/03_Classification_Benchmark.ipynb` | Main 80/20 balanced benchmark: nine classical models, TensorFlow DNN, optional five-fold CV | `benchmark.joblib`, `benchmark_metrics.csv`, `models/` |
| `notebooks/04_Ensembles_and_Feature_Ablation.ipynb` | Stacking, hard voting, probability averaging, observed-photometry ablation | `ensembles.joblib`, `ensemble_metrics.csv`, `feature_ablation.csv` |
| `notebooks/05_Class_Balancing_Sensitivity.ipynb` | Class caps 1k/2.5k/5k/10k, unbalanced training, inverse-frequency weights, ensemble comparison | `balancing_*.csv`, `catboost_balanced_sample_weights.csv` |
| `notebooks/06_SHAP_and_Interpretability.ipynb` | Fitted CatBoost SHAP values and optional ensemble permutation importance | `shap_global.csv`, `shap_classwise.csv`, figures |
| `notebooks/07_External_LAMOST_Validation_Optional.ipynb` | Optional independent LAMOST DR11 comparison on Gaia test sources | `lamost_external_metrics.csv`, figures (when matches are available) |

SHAP values are calculated by CatBoost's native TreeSHAP algorithm by default (`shap_backend="catboost_native"`), with the original `shap.TreeExplainer` API available through `shap_backend="tree_explainer"` when compatible with the installed packages.

The modules in `src/gaia_pipeline.py` implement **the actual reusable training, preprocessing, inference and metric procedures**. The notebooks call those methods and provide documented experiments. Model hyperparameters, random seed, feature caps and execution flags are defined in `config/experiment.json`.

### Full reproduction versus a quick pipeline check

`run_full_benchmark=true`, `run_cross_validation=true`, and `run_dnn=true` are the **publication-intended defaults**. Training all models and five CV folds, particularly the SVM and DNN, may take significant time and memory. To test the workflow quickly, temporarily set:

```json
"run_full_benchmark": false,
"run_cross_validation": false,
"run_dnn": false,
"run_tsne": false,
"run_balancing_ensemble": false
```

A quick run trains only the three ensemble base learners, and **does not reproduce the full ten-model benchmark**. Restore the publication-intended settings before generating final tables.

## Exact preprocessing and experimental populations

**Quality filters:** `Plx > 0`, finite `Dist > 0`, `RUWE < 1.4`, finite observed G/BP/RP magnitudes and extinctions `AG`, `ABP`, `ARP`, `Pstar > 0.9`, `PGal < 0.1`; keep the seven target labels. The modelling cohort requires nonmissing `Rad-Flame`, `Lum-Flame`, and `Mass-Flame`.

**Observed photometry:** `Gmag`, `BPmag`, `RPmag`, and their differences `BP_RP_obs`, `BP_G_obs`, `G_RP_obs`.

**Corrected photometry:** `G_abs`, `BP_abs`, `RP_abs` and their colour indices, with $M_X=m_X-[5\log_{10}(D)-5]-A_X$; distance $D$ is supplied by the catalogue, clipped to at least 1 pc. These corrected features therefore incorporate catalogue-derived distance and extinction estimates.

**Inferred parameters:** `Teff`, `logg`, `[Fe/H]`, `Rad`, `EWHa`, `Rad-Flame`, `Lum-Flame`, `Mass-Flame`. The combined feature set contains the corrected photometry plus these parameters.

**Main benchmark:** a fixed seed-42 stratified 80/20 split of the 35,000-source balanced cohort (**28,000 train / 7,000 test**). The five CV folds are limited to training data. The recorded hyperparameters are preserved in the JSON configuration. The ablation reuses exactly the same held-out test sources.

**Subsampling sensitivity:** a **separate** fixed stratified 80/20 split of the FLAME-complete cohort (**403,444 train / 100,862 test**). Only training data are subsampled. All six CatBoost settings and both hard-voting settings use the same 100,862-source held-out test set, with disjoint Gaia identifiers. The 10,000-per-class cap yields 66,532 training sources because only 6,532 O-class sources were available in the training pool. Inverse-frequency sample weights come from `compute_sample_weight(class_weight="balanced")`, i.e., $w_k=N/(K n_k)$; in the original run weights ranged from **0.7793 to 8.8235**. Unlike the main benchmark, this held-out set is imbalanced. Results from the two evaluation protocols must not be compared as if drawn from the same test distribution.

**External validation:** notebook 07 uses LAMOST DR11 `STAR` records with G-band SNR ≥ 10; repeated observations are resolved by maximum SNR and inconsistent repeat labels are reported. TAP access is disabled by default because service availability, catalog schema and match counts are external dependencies. To run offline, place a previously retrieved match file at `data/lamost_dr11_matches.csv` with `gaia_id`, `lamostobjectclass`, `lamostsubclass`, `snr_g` fields. The notebook never uses LAMOST labels for model fitting or tuning.

## Outputs, versioning, and reproducibility

- All intermediate data, trained models, cached LAMOST matches, tables and figures are written under `outputs/` or `data/` and excluded from Git via `.gitignore`.
- Publishing the source code without generated data files is intentional. Re-run with the Kaggle dataset to regenerate the full results.
- Small variations can occur with dependency versions, hardware and numerical implementation. The provided code follows the user-supplied research notebook; **the complete pipeline has not been independently re-executed on the original Kaggle CSV in this packaging environment**, because the CSV was not supplied.
- Tests in `tests/` verify essential transformations and invariants on synthetic records. Synthetic tests are not evidence of reproducing the paper's numerical metrics.

## Repository and manuscript citation

**Gudiño-Ochoa, Alberto; Ochoa-Ornelas, Raquel.** *Gaia DR3 Catalogue Spectral-Type Classification with Ensemble Learning.* Reproducibility repository associated with the Franklin Open manuscript.

**Repository URL:** **[ADD THE PUBLIC GITHUB URL AFTER CREATING THE REPOSITORY]**. Insert the final, publicly accessible URL in the manuscript's **Data Availability** section before resubmission. A public Zenodo release/DOI can optionally be added after the repository is frozen.

No license is assigned in this package; the authors should choose and add an appropriate license before publication.
