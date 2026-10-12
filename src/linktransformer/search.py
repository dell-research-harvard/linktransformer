"""Exact inner-product search over L2-normalized embeddings, implemented in torch.

This replaces the previous ``faiss.IndexFlatIP`` usage. Results match a flat
(exact) FAISS index, but only torch's OpenMP runtime is loaded, which avoids the
crashes caused by FAISS and torch each bundling their own ``libomp`` on macOS.
Queries are processed in chunks so the score matrix stays within a fixed memory
budget, and the search runs on CUDA when it is available.
"""
from typing import Iterator, Tuple

import numpy as np
import torch

# Upper bound on the number of float32 scores held in memory at once (~256 MB).
_MAX_SCORES_PER_CHUNK = 1 << 26


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).to(device)


def _query_chunks(n_queries: int, n_corpus: int) -> Iterator[Tuple[int, int]]:
    step = max(1, _MAX_SCORES_PER_CHUNK // max(n_corpus, 1))
    for start in range(0, n_queries, step):
        yield start, min(start + step, n_queries)


def knn_search(queries: np.ndarray, corpus: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Find the ``k`` corpus rows with the highest inner product for each query row.

    :param queries: (n_queries, dim) array of normalized embeddings.
    :param corpus: (n_corpus, dim) array of normalized embeddings.
    :param k: number of neighbours per query; capped at ``n_corpus``.
    :return: ``(scores, indices)``, each of shape (n_queries, min(k, n_corpus)), sorted by
        descending score. Same layout as ``faiss.IndexFlatIP.search``.
    """
    k = min(k, len(corpus))
    if len(queries) == 0 or k == 0:
        return np.empty((len(queries), k), dtype=np.float32), np.empty((len(queries), k), dtype=np.int64)

    device = _device()
    corpus_t = _as_tensor(corpus, device)
    scores, indices = [], []
    with torch.no_grad():
        for start, end in _query_chunks(len(queries), len(corpus)):
            sims = _as_tensor(queries[start:end], device) @ corpus_t.T
            top_scores, top_indices = torch.topk(sims, k, dim=1)
            scores.append(top_scores.cpu().numpy())
            indices.append(top_indices.cpu().numpy())
    return np.concatenate(scores), np.concatenate(indices)


def range_search(
    queries: np.ndarray, corpus: np.ndarray, threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find all corpus rows whose inner product with each query is at least ``threshold``.

    :param queries: (n_queries, dim) array of normalized embeddings.
    :param corpus: (n_corpus, dim) array of normalized embeddings.
    :param threshold: minimum similarity to keep.
    :return: ``(lims, scores, indices)`` in the layout of ``faiss.IndexFlatIP.range_search``:
        the matches of query ``i`` are ``scores[lims[i]:lims[i + 1]]`` and
        ``indices[lims[i]:lims[i + 1]]``, ordered by corpus index.
    """
    lims = np.zeros(len(queries) + 1, dtype=np.int64)
    if len(queries) == 0 or len(corpus) == 0:
        return lims, np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)

    device = _device()
    corpus_t = _as_tensor(corpus, device)
    scores, indices = [], []
    with torch.no_grad():
        for start, end in _query_chunks(len(queries), len(corpus)):
            sims = _as_tensor(queries[start:end], device) @ corpus_t.T
            # nonzero() returns (row, col) pairs in row-major order, i.e. grouped by
            # query and ordered by corpus index within each query.
            rows, cols = (sims >= threshold).nonzero(as_tuple=True)
            scores.append(sims[rows, cols].cpu().numpy())
            indices.append(cols.cpu().numpy())
            counts = torch.bincount(rows, minlength=end - start).cpu().numpy()
            lims[start + 1:end + 1] = counts
    np.cumsum(lims, out=lims)
    return lims, np.concatenate(scores).astype(np.float32), np.concatenate(indices).astype(np.int64)
