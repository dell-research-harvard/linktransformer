import numpy as np
import pytest

from linktransformer import search


def _normalized(rng, n, dim=16):
    x = rng.standard_normal((n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.fixture
def small_chunks(monkeypatch):
    # force several query chunks so the chunking path is exercised
    monkeypatch.setattr(search, "_MAX_SCORES_PER_CHUNK", 500)


@pytest.mark.parametrize("chunked", [False, True])
def test_lt_knn_search_matches_brute_force(request, chunked):
    if chunked:
        request.getfixturevalue("small_chunks")
    rng = np.random.default_rng(0)
    queries, corpus = _normalized(rng, 57), _normalized(rng, 120)

    scores, indices = search.knn_search(queries, corpus, k=4)

    sims = queries @ corpus.T
    expected = np.argsort(-sims, axis=1)[:, :4]
    assert np.array_equal(indices, expected)
    assert np.allclose(scores, np.take_along_axis(sims, expected, axis=1), atol=1e-6)


def test_lt_knn_search_caps_k_at_corpus_size():
    rng = np.random.default_rng(1)
    scores, indices = search.knn_search(_normalized(rng, 3), _normalized(rng, 2), k=5)
    assert indices.shape == scores.shape == (3, 2)


def test_lt_knn_search_empty_queries():
    rng = np.random.default_rng(2)
    scores, indices = search.knn_search(_normalized(rng, 0), _normalized(rng, 4), k=2)
    assert indices.shape == (0, 2)


@pytest.mark.parametrize("chunked", [False, True])
def test_lt_range_search_matches_brute_force(request, chunked):
    if chunked:
        request.getfixturevalue("small_chunks")
    rng = np.random.default_rng(3)
    queries, corpus = _normalized(rng, 41), _normalized(rng, 90)

    lims, scores, indices = search.range_search(queries, corpus, threshold=0.25)

    sims = queries @ corpus.T
    assert lims[0] == 0 and lims[-1] == len(indices) == len(scores)
    for i in range(len(queries)):
        expected = np.flatnonzero(sims[i] >= 0.25)  # ascending corpus index, like faiss
        assert np.array_equal(indices[lims[i]:lims[i + 1]], expected)
        assert np.allclose(scores[lims[i]:lims[i + 1]], sims[i, expected], atol=1e-6)


def test_lt_range_search_empty_corpus():
    rng = np.random.default_rng(4)
    lims, scores, indices = search.range_search(_normalized(rng, 3), _normalized(rng, 0), threshold=0.5)
    assert np.array_equal(lims, np.zeros(4)) and len(scores) == len(indices) == 0
