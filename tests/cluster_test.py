"""Clustering and deduplication: cluster_embeddings, cluster_rows and dedup_rows."""
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering

import linktransformer as lt
from linktransformer import cluster_fns
from linktransformer import infer as infer_mod


def _points(n, dim=16, centers=30, spread=0.25, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((centers, dim))[rng.integers(0, centers, n)] + spread * rng.standard_normal((n, dim))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def _same_partition(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return len(set(zip(a, b))) == len(set(a)) == len(set(b))


@pytest.mark.parametrize("seed", range(12))
def test_lt_slink_matches_sklearn_dbscan(seed):
    rng = np.random.default_rng(seed)
    x = _points(int(rng.integers(5, 300)), spread=float(rng.uniform(0.05, 0.6)), seed=seed)
    sim, min_size = float(rng.uniform(0.6, 0.99)), int(rng.integers(1, 6))
    weights = rng.integers(1, 4, len(x)) if seed % 2 else np.ones(len(x), dtype=np.int64)
    expected = DBSCAN(eps=1 - sim, min_samples=min_size, metric="cosine").fit(x, sample_weight=weights).labels_
    assert np.array_equal(cluster_fns._slink(x, sim, min_size, weights), expected)


@pytest.mark.parametrize("min_size", [2, 4])
def test_lt_slink_in_small_chunks_matches_one_chunk(monkeypatch, min_size):
    x = _points(1500, spread=0.3)
    expected = DBSCAN(eps=0.15, min_samples=min_size, metric="cosine").fit(x).labels_
    monkeypatch.setattr(cluster_fns, "_PAIR_BUDGET", 300)  # many range searches, growing and shrinking
    assert np.array_equal(cluster_fns._slink(x, 0.85, min_size, np.ones(len(x), dtype=np.int64)), expected)


@pytest.mark.parametrize("linkage", ["average", "complete", "single"])
def test_lt_agglomerative_sim_threshold_matches_sklearn(linkage):
    x = _points(400)
    expected = AgglomerativeClustering(n_clusters=None, distance_threshold=0.2, linkage=linkage, metric="cosine").fit(x).labels_
    labels = lt.cluster_embeddings(x, sim_threshold=0.8, cluster_type="agglomerative", cluster_params={"linkage": linkage})
    assert _same_partition(labels, expected)


def test_lt_hdbscan_uses_sklearn_and_gives_noise_own_clusters():
    x = _points(300, spread=0.4)
    expected = HDBSCAN(min_cluster_size=5, min_samples=1, copy=True).fit(x).labels_
    assert (expected == -1).any()
    labels = lt.cluster_embeddings(x, cluster_type="hdbscan")  # any capitalization
    assert np.array_equal(lt.cluster("HDBScan", None, x), expected)  # the original function keeps -1
    assert _same_partition(labels[expected >= 0], expected[expected >= 0])
    noise = labels[expected == -1]
    assert len(set(noise)) == len(noise) and not set(noise) & set(labels[expected >= 0])


def test_lt_cluster_embeddings_numbers_clusters_by_first_appearance():
    x = np.array([[0, 1], [1, 0], [0, 1], [0.99, 0.01], [-1, 0]], dtype=np.float32)
    assert lt.cluster_embeddings(x, sim_threshold=0.9).tolist() == [0, 1, 0, 1, 2]
    assert lt.cluster_embeddings(torch.tensor(x) * 3, sim_threshold=0.9).tolist() == [0, 1, 0, 1, 2]  # normalized
    assert lt.cluster_embeddings(np.empty((0, 2)), sim_threshold=0.9).tolist() == []


def test_lt_legacy_distance_threshold_equals_sim_threshold():
    x = _points(300)
    legacy = {"threshold": 0.1, "min cluster size": 2, "metric": "cosine"}
    assert np.array_equal(lt.cluster_embeddings(x, cluster_params=legacy), lt.cluster_embeddings(x, sim_threshold=0.9))
    euclidean = {"threshold": 0.4, "metric": "euclidean"}  # |a - b| <= 0.4  <=>  cos >= 1 - 0.4**2 / 2
    assert np.array_equal(lt.cluster_embeddings(x, cluster_params=euclidean), lt.cluster_embeddings(x, sim_threshold=0.92))
    # the original agglomerative defaults (ward linkage on euclidean distance) still apply with "threshold"
    expected = AgglomerativeClustering(n_clusters=None, distance_threshold=0.7, linkage="ward").fit(x).labels_
    assert _same_partition(lt.cluster_embeddings(x, cluster_type="agglomerative", cluster_params={"threshold": 0.7}), expected)


@pytest.mark.parametrize("kwargs, message", [
    (dict(), "needs a similarity threshold"),
    (dict(cluster_type="agglomerative"), "needs a similarity threshold"),
    (dict(sim_threshold=0.9, cluster_params={"threshold": 0.1}), "not both"),
    (dict(sim_threshold=0.9, cluster_params={"min_samples": 2}), "Unknown cluster_params"),
    (dict(sim_threshold=0.9, cluster_params={"treshold": 0.1}), "Unknown cluster_params"),
    (dict(sim_threshold=0.9, cluster_type="HDBScan"), "no similarity threshold"),
    (dict(sim_threshold=0.9, cluster_type="kmeans"), "cluster_type must be one of"),
    (dict(sim_threshold=1.5), "between -1 and 1"),
    (dict(sim_threshold=0.9, cluster_type="agglomerative", cluster_params={"linkage": "ward"}), "ward"),
])
def test_lt_cluster_options_are_checked(kwargs, message):
    with pytest.raises(ValueError, match=message):
        lt.cluster_embeddings(_points(20), **kwargs)


def test_lt_cluster_params_accept_both_key_styles():
    resolved = cluster_fns.resolve_cluster_params("hdbscan", None, {"min cluster size": 3, "min_samples": 2})
    assert resolved == {"cluster_type": "HDBScan", "min_cluster_size": 3, "min_samples": 2}
    resolved = cluster_fns.resolve_cluster_params("Agglomerative", 0.8, {"clustering linkage": "complete"})
    assert resolved["linkage"] == "complete" and resolved["distance_threshold"] == pytest.approx(0.2)


def test_lt_agglomerative_refuses_data_too_large_for_memory(monkeypatch):
    monkeypatch.setattr(cluster_fns, "_physical_memory", lambda: 10_000)
    with pytest.raises(MemoryError, match="SLINK"):
        lt.cluster_embeddings(_points(200), sim_threshold=0.8, cluster_type="agglomerative")


# cluster_rows / dedup_rows with embeddings looked up from a table instead of a model
_VECTORS = {
    "Apple Inc": [1.0, 0.0, 0.0], "Apple Incorporated": [0.99, 0.1, 0.0], "Apple": [0.97, 0.2, 0.0],
    "Microsoft": [0.0, 1.0, 0.0], "Toyota": [0.0, 0.0, 1.0], "Toyota Motor": [0.0, 0.1, 0.99],
    "Banana Republic": [0.5, -0.5, -0.7],
}


@pytest.fixture
def embedded(monkeypatch):
    """Texts passed to infer_embeddings, which returns vectors from _VECTORS."""
    calls = []

    def fake_infer_embeddings(strings, model, **kwargs):
        calls.append(list(strings))
        return np.array([_VECTORS[s.split(" | ")[0]] for s in strings], dtype=np.float32)

    monkeypatch.setattr(infer_mod, "infer_embeddings", fake_infer_embeddings)
    return calls


def _firms():
    return pd.DataFrame({
        "name": ["Apple Inc", "Microsoft", "Apple Incorporated", "Apple Inc", None, "Toyota", "Banana Republic",
                 "Toyota Motor", None],
        "country": ["US", "US", "UK", "US", "US", "JP", "US", "US", "JP"],
    }, index=[10, 11, 12, 13, 14, 15, 16, 17, 18])


def test_lt_cluster_rows_gives_every_row_a_cluster(embedded):
    out = lt.cluster_rows(_firms(), model=object(), on="name", sim_threshold=0.9)
    assert out["cluster"].tolist() == [0, 1, 0, 0, 2, 3, 4, 3, 5]
    assert out["cluster_size"].tolist() == [3, 1, 3, 3, 1, 2, 1, 2, 1]
    assert list(out.index) == list(_firms().index) and "cluster" not in _firms()
    # each distinct text is embedded once; missing names are not embedded
    assert len(embedded) == 1 and sorted(embedded[0]) == sorted(set(_firms()["name"].dropna()))


def test_lt_cluster_rows_within_blocks(embedded):
    out = lt.cluster_rows(_firms(), model=object(), on="name", sim_threshold=0.9, blocking_vars="country")
    # Apple Incorporated (UK) and Toyota Motor (US) have no near-duplicate in their own country
    assert out["cluster"].tolist() == [0, 1, 2, 0, 3, 4, 5, 6, 7]
    assert len(embedded) == 1  # one embedding pass for all blocks


def test_lt_cluster_rows_on_several_columns(embedded, monkeypatch):
    monkeypatch.setattr(infer_mod, "serialize_columns", lambda df, cols, model=None: [" | ".join(map(str, r)) for r in df[cols].values])
    out = lt.cluster_rows(_firms().dropna(), model=object(), on=["name", "country"], sim_threshold=0.9)
    assert out["cluster"].tolist() == [0, 1, 0, 0, 2, 3, 2]


@pytest.mark.parametrize("cluster_type, kwargs", [
    ("agglomerative", dict(sim_threshold=0.9)),
    ("HDBScan", dict(cluster_params={"min_cluster_size": 2})),
])
def test_lt_cluster_rows_identical_texts_share_a_cluster(embedded, cluster_type, kwargs):
    out = lt.cluster_rows(_firms(), model=object(), on="name", cluster_type=cluster_type, **kwargs)
    assert out["cluster"].iloc[0] == out["cluster"].iloc[3]  # both "Apple Inc"
    assert out["cluster"].iloc[4] != out["cluster"].iloc[8]  # missing names are never grouped


def test_lt_cluster_rows_checks_options_before_embedding(embedded):
    with pytest.raises(ValueError, match="needs a similarity threshold"):
        lt.cluster_rows(_firms(), model=object(), on="name")
    with pytest.raises(ValueError, match="Pass on="):
        lt.cluster_rows(_firms(), model=object(), sim_threshold=0.9)
    with pytest.raises(ValueError, match="not in the dataframe"):
        lt.cluster_rows(_firms(), model=object(), on="name", sim_threshold=0.9, blocking_vars="region")
    assert embedded == []


def test_lt_dedup_rows_keeps_one_row_per_cluster(embedded):
    out = lt.dedup_rows(_firms(), model=object(), on="name", sim_threshold=0.9)
    # unclustered rows and rows with a missing name are all kept
    assert list(out.index) == [10, 11, 14, 15, 16, 18]


def test_lt_dedup_rows_keep_medoid(embedded):
    df = pd.DataFrame({"name": ["Apple Inc", "Apple Incorporated", "Apple", "Microsoft"]})
    # "Apple Incorporated" sits between the other two
    assert lt.dedup_rows(df, model=object(), on="name", sim_threshold=0.9, keep="medoid")["name"].tolist() == [
        "Apple Incorporated", "Microsoft"]
    with pytest.raises(ValueError, match="keep must be"):
        lt.dedup_rows(df, model=object(), on="name", sim_threshold=0.9, keep="last")


def test_lt_dedup_rows_within_blocks(embedded):
    out = lt.dedup_rows(_firms(), model=object(), on="name", sim_threshold=0.9, blocking_vars=["country"])
    assert list(out.index) == [10, 11, 12, 14, 15, 16, 17, 18]
