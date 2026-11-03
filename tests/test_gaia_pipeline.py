"""Lightweight, data-free invariants for the research workflow."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gaia_pipeline import FEATURE_SETS, add_photometric_features, balanced_subsample, hard_voting_from_arrays, load_input_csv, metric_dict


def test_corrected_magnitude_and_colour():
    data = pd.DataFrame({"Dist": [10.0], "Gmag": [12.0], "BPmag": [13.0], "RPmag": [11.0], "AG": [0.2], "ABP": [0.4], "ARP": [0.1]})
    out = add_photometric_features(data)
    assert np.isclose(out["G_abs"].iloc[0], 11.8)
    assert np.isclose(out["BP_RP_obs"].iloc[0], 2.0)
    assert np.isclose(out["BP_RP_abs"].iloc[0], 1.7)
    assert len(FEATURE_SETS["combined_features"]) == 14


def test_subsample_is_seeded_and_capped():
    data = pd.DataFrame({"SpType-ELS": ["O"] * 20 + ["G"] * 20, "Source": np.arange(40)})
    first = balanced_subsample(data, 5, 42)
    second = balanced_subsample(data, 5, 42)
    assert len(first) == 10
    pd.testing.assert_frame_equal(first, second)
    assert first["SpType-ELS"].value_counts().eq(5).all()


def test_hard_voting_tie_breaker():
    prediction = [np.array([0, 1]), np.array([1, 1]), np.array([2, 0])]
    probabilities = [np.array([[0.2, 0.4, 0.4], [0.2, 0.7, 0.1]]), np.array([[0.2, 0.5, 0.3], [0.1, 0.8, 0.1]]), np.array([[0.1, 0.3, 0.6], [0.5, 0.4, 0.1]])]
    label, vote_fraction = hard_voting_from_arrays(prediction, probabilities, 3)
    assert list(label) == [2, 1]
    assert np.allclose(vote_fraction.sum(axis=1), 1)


def test_metrics_are_finite():
    score = metric_dict(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    assert all(np.isfinite(v) for v in score.values())


def test_precision_of_source_id_is_preserved(tmp_path):
    from src.gaia_pipeline import REQUIRED_COLUMNS
    data = {column: [1] for column in REQUIRED_COLUMNS}
    data["Source"] = ["6052821195539871232"]
    data["SpType-ELS"] = ["O"]
    path = tmp_path / "gaia.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    loaded = load_input_csv(path)
    assert loaded["Source"].iloc[0] == 6052821195539871232
