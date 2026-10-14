import logging

import numpy as np
import pytest
import torch

from linktransformer import search


def _normalized(rng, n, dim=24):
    x = rng.standard_normal((n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _with_near_duplicates(rng, n, dim=24):
    """Normalized rows where many are exact or near duplicates of others, so top-k scores are tied or very close."""
    base = _normalized(rng, n // 3, dim)
    x = np.concatenate([base, base, base + 1e-4 * rng.standard_normal(base.shape).astype(np.float32)])
    x = np.concatenate([x, _normalized(rng, n - len(x), dim)])
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


# Search configurations; "small tiles" splits even small inputs into several query and corpus tiles
# and makes the block-max top-k path run.
CONFIGS = {
    "tiled": {},
    "small tiles": {"tiles": True},
    "untiled": {"tiled": False},
    "fp16": {"fp16": True, "tiles": True},
    "fp16 forced fallback": {"fp16": True, "tiles": True, "extra": 0},
}


@pytest.fixture(params=sorted(CONFIGS))
def config(request, monkeypatch):
    cfg = CONFIGS[request.param]
    if cfg.get("tiles"):
        monkeypatch.setattr(search, "_CPU_TILE", (7, 32))
        monkeypatch.setattr(search, "_BLOCK", 4)
    if cfg.get("tiled") is False:
        monkeypatch.setattr(search, "_MAX_SCORES_PER_CHUNK", 500)
    if cfg.get("fp16"):  # run the half-precision path on CPU
        monkeypatch.setattr(search, "_fp16_available", lambda device: True)
    if "extra" in cfg:
        monkeypatch.setattr(search, "_FP16_EXTRA_CANDIDATES", cfg["extra"])
    return {"precision": "fp16" if cfg.get("fp16") else "fp32", "tiled": cfg.get("tiled", True)}


def _check_knn(queries, corpus, k, scores, indices):
    sims = queries.astype(np.float64) @ corpus.astype(np.float64).T
    expected_scores = -np.sort(-sims, axis=1)[:, :k]
    assert indices.shape == scores.shape == (len(queries), min(k, len(corpus)))
    assert scores.dtype == np.float32 and indices.dtype == np.int64
    # Correct neighbours: the returned rows have the top-k true scores (ties may come back in any order).
    assert np.allclose(np.take_along_axis(sims, indices, axis=1), expected_scores, atol=1e-5)
    assert np.allclose(scores, expected_scores, atol=1e-5)
    assert all(len(set(row)) == len(row) for row in indices)


@pytest.mark.parametrize("k", [1, 4, 10])
def test_lt_knn_search_matches_brute_force(config, k):
    rng = np.random.default_rng(0)
    queries, corpus = _normalized(rng, 57), _normalized(rng, 150)
    scores, indices = search.knn_search(queries, corpus, k=k, **config)
    _check_knn(queries, corpus, k, scores, indices)
    sims = queries.astype(np.float64) @ corpus.astype(np.float64).T
    assert np.array_equal(indices, np.argsort(-sims, axis=1)[:, :k])  # no ties in random data


@pytest.mark.parametrize("k", [1, 5])
def test_lt_knn_search_with_near_duplicates(config, k):
    rng = np.random.default_rng(1)
    corpus = _with_near_duplicates(rng, 120)
    queries = np.concatenate([corpus[:20], _normalized(rng, 20)])
    scores, indices = search.knn_search(queries, corpus, k=k, **config)
    _check_knn(queries, corpus, k, scores, indices)


def test_lt_knn_search_unnormalized(config):
    rng = np.random.default_rng(2)
    queries = _normalized(rng, 30) * rng.uniform(0.1, 20, (30, 1)).astype(np.float32)
    corpus = _normalized(rng, 90) * rng.uniform(0.1, 20, (90, 1)).astype(np.float32)
    scores, indices = search.knn_search(queries, corpus, k=3, **config)
    _check_knn(queries, corpus, 3, scores, indices)


def test_lt_knn_search_caps_k_at_corpus_size(config):
    rng = np.random.default_rng(3)
    queries, corpus = _normalized(rng, 3), _normalized(rng, 2)
    scores, indices = search.knn_search(queries, corpus, k=5, **config)
    _check_knn(queries, corpus, 5, scores, indices)


def test_lt_knn_search_empty_inputs(config):
    rng = np.random.default_rng(4)
    scores, indices = search.knn_search(_normalized(rng, 0), _normalized(rng, 4), k=2, **config)
    assert indices.shape == scores.shape == (0, 2)
    scores, indices = search.knn_search(_normalized(rng, 3), _normalized(rng, 0), k=2, **config)
    assert indices.shape == scores.shape == (3, 0)


def _check_range(queries, corpus, threshold, lims, scores, indices):
    sims = queries.astype(np.float64) @ corpus.astype(np.float64).T
    assert np.abs(sims - threshold).min() > 1e-5, "test data should not sit on the threshold"
    assert lims[0] == 0 and lims[-1] == len(indices) == len(scores) and len(lims) == len(queries) + 1
    for i in range(len(queries)):
        expected = np.flatnonzero(sims[i] >= threshold)  # ascending corpus index, like faiss
        assert np.array_equal(indices[lims[i]:lims[i + 1]], expected)
        assert np.allclose(scores[lims[i]:lims[i + 1]], sims[i, expected], atol=1e-5)


def test_lt_range_search_matches_brute_force(config):
    rng = np.random.default_rng(5)
    queries, corpus = _normalized(rng, 41), _normalized(rng, 150)
    _check_range(queries, corpus, 0.25, *search.range_search(queries, corpus, threshold=0.25, **config))


def test_lt_range_search_with_near_duplicates(config):
    rng = np.random.default_rng(6)
    corpus = _with_near_duplicates(rng, 120)
    queries = np.concatenate([corpus[:20], _normalized(rng, 20)])
    _check_range(queries, corpus, 0.3, *search.range_search(queries, corpus, threshold=0.3, **config))


def test_lt_range_search_empty_inputs(config):
    rng = np.random.default_rng(7)
    lims, scores, indices = search.range_search(_normalized(rng, 3), _normalized(rng, 0), threshold=0.5, **config)
    assert np.array_equal(lims, np.zeros(4)) and len(scores) == len(indices) == 0
    lims, scores, indices = search.range_search(_normalized(rng, 0), _normalized(rng, 5), threshold=0.5, **config)
    assert np.array_equal(lims, np.zeros(1)) and len(scores) == len(indices) == 0


def test_lt_fp16_forced_fallback_reruns_queries_in_fp32(monkeypatch, caplog):
    monkeypatch.setattr(search, "_fp16_available", lambda device: True)
    monkeypatch.setattr(search, "_FP16_EXTRA_CANDIDATES", 0)
    rng = np.random.default_rng(8)
    queries, corpus = _normalized(rng, 20), _normalized(rng, 60)
    with caplog.at_level(logging.DEBUG, logger="linktransformer.search"):
        scores, indices = search.knn_search(queries, corpus, k=3, precision="fp16")
    _check_knn(queries, corpus, 3, scores, indices)
    assert "searched again in fp32" in caplog.text


def test_lt_fp16_on_cpu_warns_and_searches_in_fp32(monkeypatch):
    monkeypatch.setattr(search, "_device", lambda: torch.device("cpu"))
    rng = np.random.default_rng(9)
    queries, corpus = _normalized(rng, 10), _normalized(rng, 30)
    with pytest.warns(UserWarning, match="needs a CUDA GPU"):
        scores, indices = search.knn_search(queries, corpus, k=2, precision="fp16")
    _check_knn(queries, corpus, 2, scores, indices)


def test_lt_fp16_with_huge_embeddings_uses_fp32(monkeypatch):
    monkeypatch.setattr(search, "_fp16_available", lambda device: True)
    rng = np.random.default_rng(10)
    queries, corpus = _normalized(rng, 10) * 1000, _normalized(rng, 30) * 1000
    with pytest.warns(UserWarning, match="too large for half precision"):
        scores, indices = search.knn_search(queries, corpus, k=2, precision="fp16")
    _check_knn(queries, corpus, 2, scores, indices)


@pytest.mark.parametrize("call", [
    lambda: search.knn_search(np.zeros((1, 2), np.float32), np.zeros((1, 2), np.float32), 1, precision="bf16"),
    lambda: search.range_search(np.zeros((1, 2), np.float32), np.zeros((1, 2), np.float32), 0.5, tiled="yes"),
])
def test_lt_search_rejects_bad_options(call):
    with pytest.raises(ValueError):
        call()


def test_lt_search_disables_tf32_and_restores_setting():
    rng = np.random.default_rng(11)
    queries, corpus = _normalized(rng, 10), _normalized(rng, 30)
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        seen = []
        original = search._matmul_fp32
        search._matmul_fp32 = lambda q, c: (seen.append(torch.get_float32_matmul_precision()), original(q, c))[1]
        try:
            scores, indices = search.knn_search(queries, corpus, k=2)
        finally:
            search._matmul_fp32 = original
        assert seen and set(seen) == {"highest"}
        assert torch.get_float32_matmul_precision() == "high"
        _check_knn(queries, corpus, 2, scores, indices)
    finally:
        torch.set_float32_matmul_precision(previous)
