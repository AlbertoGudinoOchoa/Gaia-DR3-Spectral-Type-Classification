"""Shared, transparent methods for Gaia DR3 catalogue spectral-type classification.

This module follows the source research notebook supplied by the authors.
Predictions are evaluated against catalogue-derived SpType-ELS labels.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

REQUIRED_COLUMNS = ["Source", "RA_ICRS", "DE_ICRS", "Plx", "Dist", "RUWE", "Gmag", "BPmag", "RPmag", "AG", "ABP", "ARP", "Pstar", "PGal", "SpType-ELS", "Teff", "logg", "[Fe/H]", "Rad", "EWHa", "Rad-Flame", "Lum-Flame", "Mass-Flame"]
FLAME_COLUMNS = ["Rad-Flame", "Lum-Flame", "Mass-Flame"]
OBSERVED_FEATURES = ["Gmag", "BPmag", "RPmag", "BP_RP_obs", "BP_G_obs", "G_RP_obs"]
CORRECTED_FEATURES = ["G_abs", "BP_abs", "RP_abs", "BP_RP_abs", "BP_G_abs", "G_RP_abs"]
INFERRED_FEATURES = ["Teff", "logg", "[Fe/H]", "Rad", "EWHa", "Rad-Flame", "Lum-Flame", "Mass-Flame"]
FEATURE_SETS = {"photometric_features": CORRECTED_FEATURES, "inferred_astrophysical_parameters": INFERRED_FEATURES, "combined_features": CORRECTED_FEATURES + INFERRED_FEATURES}
BASE_MODELS = ("CatBoost", "MLP", "LightGBM")
ENSEMBLE_NAMES = {"stacking": "Stacking_CatBoost_MLP_LightGBM", "hard_voting": "HardVoting_CatBoost_MLP_LightGBM", "probability_average": "ProbabilityAverage_CatBoost_MLP_LightGBM"}


def find_root() -> Path:
    """Find the repository from the current notebook working directory."""
    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "config" / "experiment.json").is_file():
            return candidate
    raise FileNotFoundError("Run Jupyter from the repository or its notebooks/ folder.")


def load_config(root: Path | None = None) -> dict:
    root = root or find_root()
    with (root / "config" / "experiment.json").open(encoding="utf-8") as file:
        return json.load(file)


def output_dir(root: Path | None = None) -> Path:
    path = (root or find_root()) / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_input_csv(path: Path) -> pd.DataFrame:
    """Load Gaia Source IDs as strings first to avoid floating-point precision loss."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing Gaia input CSV: {path}. See README.md for dataset setup.")
    data = pd.read_csv(path, dtype={"Source": "string", "SpType-ELS": "string"}, low_memory=False)
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required Gaia columns: {missing}")
    identifiers = data["Source"].str.strip()
    if identifiers.isna().any() or not identifiers.str.fullmatch(r"\d+").all():
        raise ValueError("Gaia source IDs must be non-null decimal integers; avoid float-exported IDs.")
    data["Source"] = identifiers.astype("int64")
    if data["Source"].duplicated().any():
        raise ValueError("Input contains duplicate Gaia Source IDs.")
    data["SpType-ELS"] = data["SpType-ELS"].str.strip()
    for column in REQUIRED_COLUMNS:
        if column not in ("Source", "SpType-ELS"):
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def quality_cut_gaia(data: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply the original source notebook's Gaia quality rules."""
    q = cfg["quality_filters"]
    mask = data["SpType-ELS"].isin(cfg["spectral_classes"])
    mask &= (data["Plx"] > 0) & (data["Dist"] > 0) & np.isfinite(data["Dist"])
    mask &= data["RUWE"] < q["ruwe_max"]
    for column in ("Gmag", "BPmag", "RPmag", "AG", "ABP", "ARP"):
        mask &= np.isfinite(data[column])
    mask &= (data["Pstar"] > q["pstar_min"]) & (data["PGal"] < q["pgal_max"])
    return data.loc[mask].reset_index(drop=True)


def add_photometric_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute observed colours and extinction-corrected absolute magnitudes."""
    data = data.copy()
    modulus = 5 * np.log10(data["Dist"].clip(lower=1.0)) - 5
    data["BP_RP_obs"] = data["BPmag"] - data["RPmag"]
    data["BP_G_obs"] = data["BPmag"] - data["Gmag"]
    data["G_RP_obs"] = data["Gmag"] - data["RPmag"]
    for band, column in (("G", "Gmag"), ("BP", "BPmag"), ("RP", "RPmag")):
        data[f"{band}_abs"] = data[column] - modulus - data[f"A{band}"]
    data["BP_RP_abs"] = data["BP_abs"] - data["RP_abs"]
    data["BP_G_abs"] = data["BP_abs"] - data["G_abs"]
    data["G_RP_abs"] = data["G_abs"] - data["RP_abs"]
    return data


def balanced_subsample(data: pd.DataFrame, cap: int, seed: int = 42) -> pd.DataFrame:
    """Match the original group's independent seeded random subsampling."""
    groups = [group.sample(n=min(cap, len(group)), random_state=seed) for _, group in data.groupby("SpType-ELS", sort=True)]
    return pd.concat(groups, ignore_index=True)


def split_indices(data: pd.DataFrame, cfg: dict):
    return train_test_split(np.arange(len(data)), test_size=cfg["test_size"], stratify=data["SpType-ELS"], random_state=cfg["random_state"])


def make_model(name: str, cfg: dict):
    """Construct a classifier using the authors' recorded hyperparameters."""
    h = cfg["hyperparameters"][name].copy()
    seed = cfg["random_state"]
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**h, random_seed=seed, verbose=False, allow_writing_files=False)
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**h, random_state=seed, verbosity=-1)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(**h, random_state=seed)
    if name == "DecisionTree":
        return DecisionTreeClassifier(**h, random_state=seed)
    if name == "GradientBoosting":
        return GradientBoostingClassifier(**h, random_state=seed)
    if name == "SVM":
        return SVC(**h, random_state=seed)
    if name == "RandomForest":
        return RandomForestClassifier(**h, random_state=seed)
    if name == "KNN":
        return KNeighborsClassifier(**h)
    if name == "MLP":
        return MLPClassifier(**h, random_state=seed)
    raise KeyError(f"Unknown classifier: {name}")



class ModelPipeline(Pipeline):
    """Preserve CatBoost fit-state checks across scikit-learn API changes."""
    def __sklearn_is_fitted__(self):
        final = self.steps[-1][1]
        if hasattr(final, "is_fitted"):
            return final.is_fitted()
        from sklearn.utils.validation import check_is_fitted
        try:
            check_is_fitted(final)
            return True
        except Exception:
            return False


def make_base_pipeline(name: str, cfg: dict) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if name in {"SVM", "KNN", "MLP"}:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", clone(make_model(name, cfg))))
    return ModelPipeline(steps)


def build_dnn(input_dim: int, n_classes: int, cfg: dict):
    """Implement the exact 256/128/64 architecture in the source notebook."""
    import tensorflow as tf
    d = cfg["hyperparameters"]["DNN"]
    network = tf.keras.Sequential([tf.keras.Input(shape=(input_dim,))])
    for units, dropout in zip(d["hidden_units"], d["dropout"]):
        network.add(tf.keras.layers.Dense(units, activation="relu"))
        network.add(tf.keras.layers.BatchNormalization())
        network.add(tf.keras.layers.Dropout(dropout))
    network.add(tf.keras.layers.Dense(n_classes, activation="softmax"))
    network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=d["learning_rate"]), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return network


def metric_dict(y_true, y_pred) -> dict:
    y_pred = np.asarray(y_pred).ravel().astype(int)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0), "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0), "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0), "mcc": matthews_corrcoef(y_true, y_pred)}


def hard_voting_from_arrays(predictions, probabilities, n_classes: int):
    """Majority vote with mean posterior as a tie-breaker; return vote fractions."""
    labels = np.stack([np.asarray(p).ravel().astype(int) for p in predictions])
    mean_proba = np.mean(np.stack(probabilities), axis=0)
    votes = np.stack([np.bincount(labels[:, i], minlength=n_classes) for i in range(labels.shape[1])])
    tied = votes == votes.max(axis=1, keepdims=True)
    selected = np.where(tied, mean_proba, -np.inf).argmax(axis=1)
    return selected.astype(int), votes / len(predictions)


def normalize_multiclass_shap(values, n_classes: int):
    if isinstance(values, list):
        values = np.stack(values, axis=2)
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[2] == n_classes:
        return values
    if values.ndim == 3 and values.shape[0] == n_classes:
        return np.transpose(values, (1, 2, 0))
    raise ValueError(f"Unexpected multiclass SHAP shape: {values.shape}")


def save_table(table: pd.DataFrame, root: Path, filename: str) -> Path:
    path = output_dir(root) / filename
    table.to_csv(path, index=False)
    return path


def save_object(obj, root: Path, filename: str) -> Path:
    path = output_dir(root) / filename
    joblib.dump(obj, path)
    return path


def read_object(root: Path, filename: str):
    path = root / "outputs" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required result {path} is missing. Run preceding notebooks in README order.")
    return joblib.load(path)
