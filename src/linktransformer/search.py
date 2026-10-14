"""Exact inner-product search over embeddings, implemented in torch.

This replaces the previous ``faiss.IndexFlatIP`` usage. Results match a flat (exact)
FAISS index, but only torch's OpenMP runtime is loaded, which avoids the crashes caused
by FAISS and torch each bundling their own ``libomp`` on macOS. The search runs on CUDA
when it is available.

How the search is organized:

- Tiling (``tiled=True``, the default): queries and corpus are both split into tiles, so
  every matrix product is large and roughly square and only one tile of scores is held in
  memory at a time. Each query keeps a running top-k across the corpus tiles.
  ``tiled=False`` multiplies chunks of queries against the whole corpus at once, as in
  LinkTransformer 0.2.0 before tiling was added.
- Top-k within a tile uses block maxima: the k highest scores of a row lie in the k blocks
  of ``_BLOCK`` (CPU) or ``_CUDA_BLOCK`` (CUDA) columns with the highest maxima, so only
  those blocks are ranked. This is exact and much cheaper than ranking every column.
- ``precision="fp16"`` (CUDA only): scores are first computed in half precision to keep
  ``k + _FP16_EXTRA_CANDIDATES`` candidates per query, which are then rescored in fp32.
  Half-precision rounding error has a worst-case bound, so for each query we can check
  that no row outside the candidates can reach the top k; queries that fail the check are
  searched again in fp32. Range search keeps half-precision scores above
  ``threshold - bound`` and rechecks them in fp32. Either way the results are those of an
  fp32 search. On CPU, ``"fp16"`` runs in fp32.

As with FAISS, rows with exactly equal scores (for example duplicate rows) may be returned
in any order, and which of them is returned can differ between devices and settings.
"""
import contextlib
import functools
import logging
import math
import warnings
from typing import Iterator, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

PRECISIONS = ("fp32", "fp16")

# Tile shapes (queries x corpus rows). On CPU this is about 1 GB of fp32 scores.
_CPU_TILE = (8192, 32768)
# On CUDA the corpus tile is fixed and the query tile uses a share of the free GPU memory.
_CUDA_CORPUS_TILE = 65536
_CUDA_MAX_SCORES = 1 << 30
_CUDA_FREE_MEMORY_SHARE = 0.25
# Without tiling: upper bound on the number of float32 scores held at once (~256 MB).
_MAX_SCORES_PER_CHUNK = 1 << 26
# Columns per block for the block-max top-k (CPU, CUDA).
_BLOCK = 64
_CUDA_BLOCK = 128
# Extra half-precision candidates per query, rescored in fp32.
_FP16_EXTRA_CANDIDATES = 16
# Half precision overflows above 65504; beyond this score magnitude fp32 is used instead.
_FP16_MAX_SCORE = 1e4
# Upper bound on the number of float32 values gathered at once when rescoring candidates.
_RESCORE_CHUNK = 1 << 26

_U16 = 2.0 ** -11  # unit roundoff of float16
_U32 = 2.0 ** -24  # unit roundoff of float32


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fp16_available(device: torch.device) -> bool:
    """Half-precision search is only faster on CUDA (tensor cores)."""
    return device.type == "cuda"


def check_search_options(precision: str, tiled: bool) -> None:
    """Raise ``ValueError`` for an unknown ``precision`` or a non-boolean ``tiled``."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    if not isinstance(tiled, (bool, np.bool_)):
        raise ValueError(f"tiled must be True or False, got {tiled!r}")


def _as_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).to(device)


@contextlib.contextmanager
def _full_fp32_matmul() -> Iterator[None]:
    """Disable TF32 (and similar reduced-precision fp32 matmuls) while searching."""
    previous = torch.get_float32_matmul_precision()
    if previous != "highest":
        torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        if previous != "highest":
            torch.set_float32_matmul_precision(previous)


def _tile_shape(n_corpus: int, device: torch.device, tiled: bool) -> Tuple[int, int]:
    """(query rows, corpus rows) per tile."""
    n_corpus = max(n_corpus, 1)
    if not tiled:
        return max(1, _MAX_SCORES_PER_CHUNK // n_corpus), n_corpus
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        budget = int(min(_CUDA_MAX_SCORES, max(1 << 22, free * _CUDA_FREE_MEMORY_SHARE // 4)))
        corpus_tile = min(n_corpus, _CUDA_CORPUS_TILE)
    else:
        budget = _CPU_TILE[0] * _CPU_TILE[1]
        corpus_tile = min(n_corpus, _CPU_TILE[1])
    return max(1, budget // corpus_tile), corpus_tile


def _topk(scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact top-k of each row, sorted by descending score."""
    rows, n = scores.shape
    if k == 1:
        return scores.max(dim=1, keepdim=True)
    block = _CUDA_BLOCK if scores.is_cuda else _BLOCK
    if n % block == 0 and n // block > k:
        # The top k entries lie in the k blocks with the largest maxima: any entry outside them
        # is beaten by the maxima of k other blocks.
        blocks = torch.topk(scores.view(rows, n // block, block).amax(dim=2), k, dim=1).indices
        cols = (blocks.unsqueeze(-1) * block + torch.arange(block, device=scores.device)).reshape(rows, k * block)
        values, pos = torch.topk(torch.gather(scores, 1, cols), k, dim=1)
        return values, torch.gather(cols, 1, pos)
    return torch.topk(scores, min(k, n), dim=1)


def _matmul_fp32(queries: torch.Tensor, corpus: torch.Tensor) -> torch.Tensor:
    return queries @ corpus.T


@functools.lru_cache(maxsize=None)
def _fp16_output_is_fp32(device_type: str) -> bool:
    """Whether half-precision products can be returned in fp32 (``torch.mm(out_dtype=...)``)."""
    ones = torch.ones(2, 2, dtype=torch.float16, device=device_type)
    try:
        torch.mm(ones, ones, out_dtype=torch.float32)
    except (TypeError, RuntimeError):
        return False
    return True


def _matmul_fp16(queries: torch.Tensor, corpus: torch.Tensor) -> torch.Tensor:
    if _fp16_output_is_fp32(queries.device.type):
        return torch.mm(queries, corpus.T, out_dtype=torch.float32)
    return (queries @ corpus.T).float()


def _scan(queries: torch.Tensor, corpus: torch.Tensor, k: int, corpus_tile: int, matmul) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-k (scores, corpus indices) of each query over the whole corpus, one corpus tile at a time."""
    best_scores = best_indices = None
    for start in range(0, corpus.shape[0], corpus_tile):
        scores, indices = _topk(matmul(queries, corpus[start:start + corpus_tile]), k)
        indices = indices + start
        if best_scores is not None:  # merge with the best so far
            scores, indices = torch.cat([best_scores, scores], 1), torch.cat([best_indices, indices], 1)
            scores, pos = _topk(scores, k)
            indices = torch.gather(indices, 1, pos)
        best_scores, best_indices = scores, indices
    return best_scores, best_indices


def _pair_scores(queries: torch.Tensor, corpus: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """fp32 inner products of ``queries[rows]`` and ``corpus[cols]``, elementwise so no reduced-precision matmul applies."""
    out = torch.empty(rows.shape, dtype=torch.float32, device=queries.device)
    flat_rows, flat_cols, flat_out = rows.reshape(-1), cols.reshape(-1), out.view(-1)
    step = max(1, _RESCORE_CHUNK // queries.shape[1])
    for start in range(0, flat_rows.numel(), step):
        stop = start + step
        flat_out[start:stop] = (queries[flat_rows[start:stop]] * corpus[flat_cols[start:stop]]).sum(-1)
    return out


class _HalfPrecision:
    """Half-precision copy of the corpus plus the worst-case error bound of its scores."""

    def __init__(self, corpus: torch.Tensor):
        self.corpus = corpus.half()
        self.corpus_max_norm = float(torch.linalg.vector_norm(corpus, dim=1).max()) if len(corpus) else 0.0
        self.dim = corpus.shape[1]
        # Rounding both inputs costs at most 2u * |q||c|; a half-precision output adds u * |q||c|;
        # fp32 accumulation (and the fp32 rescoring we compare against) adds about 2d * u32 * |q||c|.
        output_terms = 2 if _fp16_output_is_fp32(corpus.device.type) else 3
        self.relative_bound = output_terms * _U16 + _U16 ** 2 + (2 * self.dim + 2) * _U32

    def error_bound(self, queries: torch.Tensor) -> torch.Tensor:
        """Per-query bound on |half-precision score - true score| over all corpus rows."""
        norms = torch.linalg.vector_norm(queries, dim=1)
        # Components below the smallest normal float16 are rounded to absolute steps of 2^-25.
        underflow = 2.0 ** -25 * math.sqrt(self.dim) * (norms + self.corpus_max_norm)
        return 1.1 * (self.relative_bound * norms * self.corpus_max_norm + underflow)


def _half_precision(precision: str, queries: np.ndarray, corpus: torch.Tensor) -> Optional[_HalfPrecision]:
    """The half-precision state when ``precision="fp16"`` can be used, else None (fp32)."""
    if precision != "fp16":
        return None
    if not _fp16_available(corpus.device):
        warnings.warn("precision='fp16' needs a CUDA GPU; searching in fp32 instead.", UserWarning, stacklevel=3)
        return None
    half = _HalfPrecision(corpus)
    largest_query = float(np.linalg.norm(queries, axis=1).max()) if len(queries) else 0.0
    if largest_query * half.corpus_max_norm > _FP16_MAX_SCORE:
        warnings.warn("Embeddings are too large for half precision; searching in fp32 instead.", UserWarning, stacklevel=3)
        return None
    return half


def knn_search(
    queries: np.ndarray, corpus: np.ndarray, k: int, *, precision: str = "fp32", tiled: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Find the ``k`` corpus rows with the highest inner product for each query row.

    :param queries: (n_queries, dim) array of embeddings (normally L2-normalized).
    :param corpus: (n_corpus, dim) array of embeddings.
    :param k: number of neighbours per query; capped at ``n_corpus``.
    :param precision: ``"fp32"`` (default) or ``"fp16"``. ``"fp16"`` shortlists candidates in
        half precision on CUDA and rescores them in fp32, with a check that guarantees the same
        neighbours as ``"fp32"``. It is faster on GPUs with tensor cores; on CPU it runs in fp32.
    :param tiled: split queries and corpus into tiles (default). ``False`` multiplies query
        chunks against the whole corpus at once.
    :return: ``(scores, indices)``, each of shape (n_queries, min(k, n_corpus)), sorted by
        descending score. Same layout as ``faiss.IndexFlatIP.search``.
    """
    check_search_options(precision, tiled)
    k = min(k, len(corpus))
    if len(queries) == 0 or k == 0:
        return np.empty((len(queries), k), dtype=np.float32), np.empty((len(queries), k), dtype=np.int64)

    device = _device()
    scores, indices, fallback_rows = [], [], 0
    with torch.no_grad(), _full_fp32_matmul():
        corpus_t = _as_tensor(corpus, device)
        half = _half_precision(precision, queries, corpus_t)
        query_tile, corpus_tile = _tile_shape(len(corpus), device, tiled)
        candidates = min(k + _FP16_EXTRA_CANDIDATES, len(corpus))
        for start in range(0, len(queries), query_tile):
            query_t = _as_tensor(queries[start:start + query_tile], device)
            if half is None:
                top_scores, top_indices = _scan(query_t, corpus_t, k, corpus_tile, _matmul_fp32)
            else:
                low_scores, shortlist = _scan(query_t.half(), half.corpus, candidates, corpus_tile, _matmul_fp16)
                top_scores, pos = torch.topk(_pair_scores(query_t, corpus_t, torch.arange(len(query_t), device=device)
                                                          .unsqueeze(1).expand_as(shortlist), shortlist), k, dim=1)
                top_indices = torch.gather(shortlist, 1, pos)
                if candidates < len(corpus):
                    # Every row outside the shortlist scored at most the last candidate in half precision,
                    # so its true score is at most that plus the error bound.
                    unsure = (low_scores[:, -1] + half.error_bound(query_t) >= top_scores[:, -1]).nonzero(as_tuple=True)[0]
                    if len(unsure):
                        fallback_rows += len(unsure)
                        top_scores[unsure], top_indices[unsure] = _scan(query_t[unsure], corpus_t, k, corpus_tile, _matmul_fp32)
            scores.append(top_scores.cpu().numpy())
            indices.append(top_indices.cpu().numpy())
    if fallback_rows:
        logger.debug("fp16 knn search: %d of %d queries were searched again in fp32", fallback_rows, len(queries))
    return np.concatenate(scores).astype(np.float32), np.concatenate(indices).astype(np.int64)


def range_search(
    queries: np.ndarray, corpus: np.ndarray, threshold: float, *, precision: str = "fp32", tiled: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find all corpus rows whose inner product with each query is at least ``threshold``.

    :param queries: (n_queries, dim) array of embeddings (normally L2-normalized).
    :param corpus: (n_corpus, dim) array of embeddings.
    :param threshold: minimum similarity to keep.
    :param precision: ``"fp32"`` (default) or ``"fp16"``; see :func:`knn_search`. With ``"fp16"``,
        every half-precision match within the error bound of the threshold is rechecked in fp32.
    :param tiled: split queries and corpus into tiles (default); see :func:`knn_search`.
    :return: ``(lims, scores, indices)`` in the layout of ``faiss.IndexFlatIP.range_search``:
        the matches of query ``i`` are ``scores[lims[i]:lims[i + 1]]`` and
        ``indices[lims[i]:lims[i + 1]]``, ordered by corpus index.
    """
    check_search_options(precision, tiled)
    lims = np.zeros(len(queries) + 1, dtype=np.int64)
    if len(queries) == 0 or len(corpus) == 0:
        return lims, np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)

    device = _device()
    scores, indices = [], []
    with torch.no_grad(), _full_fp32_matmul():
        corpus_t = _as_tensor(corpus, device)
        half = _half_precision(precision, queries, corpus_t)
        query_tile, corpus_tile = _tile_shape(len(corpus), device, tiled)
        for start in range(0, len(queries), query_tile):
            query_t = _as_tensor(queries[start:start + query_tile], device)
            if half is not None:
                query_half, lower = query_t.half(), threshold - half.error_bound(query_t).unsqueeze(1)
            rows, cols, values = [], [], []
            for corpus_start in range(0, len(corpus), corpus_tile):
                corpus_slice = slice(corpus_start, corpus_start + corpus_tile)
                if half is None:
                    sims = _matmul_fp32(query_t, corpus_t[corpus_slice])
                    # nonzero() returns (row, col) pairs in row-major order, i.e. grouped by
                    # query and ordered by corpus index within each query.
                    r, c = (sims >= threshold).nonzero(as_tuple=True)
                    v = sims[r, c]
                else:
                    r, c = (_matmul_fp16(query_half, half.corpus[corpus_slice]) >= lower).nonzero(as_tuple=True)
                    v = _pair_scores(query_t, corpus_t, r, c + corpus_start)
                    keep = v >= threshold
                    r, c, v = r[keep], c[keep], v[keep]
                rows.append(r); cols.append(c + corpus_start); values.append(v)
            r, c, v = torch.cat(rows), torch.cat(cols), torch.cat(values)
            if len(rows) > 1:  # regroup by query, keeping corpus order within each query
                order = torch.sort(r, stable=True).indices
                r, c, v = r[order], c[order], v[order]
            scores.append(v.cpu().numpy())
            indices.append(c.cpu().numpy())
            lims[start + 1:start + 1 + len(query_t)] = torch.bincount(r, minlength=len(query_t)).cpu().numpy()
    np.cumsum(lims, out=lims)
    return lims, np.concatenate(scores).astype(np.float32), np.concatenate(indices).astype(np.int64)
