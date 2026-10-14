"""Clustering of embeddings, used by cluster_rows and dedup_rows.

Three methods are supported:

- "SLINK": single linkage at a similarity threshold. Two rows are linked when their cosine
  similarity is at least ``sim_threshold``, and clusters are the connected groups of linked rows.
  It runs on LinkTransformer's exact torch search (on GPU when available) and keeps only one
  chunk of matching pairs in memory, so it scales to millions of rows. With ``min_cluster_size``
  above 2 it follows DBSCAN (``min_samples=min_cluster_size``) instead.
- "agglomerative": hierarchical clustering with scikit-learn. With ``sim_threshold`` it uses
  average linkage on cosine distance: clusters are merged while their average pairwise
  similarity is above the threshold. It needs memory quadratic in the number of rows.
- "HDBScan": density-based clustering with scikit-learn's HDBSCAN. It has no threshold.

Every row gets a cluster id. Rows without a near neighbour are clusters of their own.
"""
import math
import os
from itertools import combinations
from typing import Any, Dict, Optional, Tuple

import numpy as np

CLUSTER_TYPES = ("SLINK", "agglomerative", "HDBScan")

# cluster_params keys per method; the spaced names are the original ones and are still accepted
_PARAM_ALIASES = {"min cluster size": "min_cluster_size", "min samples": "min_samples", "clustering linkage": "linkage"}
_ALLOWED_PARAMS = {
    "SLINK": {"threshold", "min_cluster_size", "metric"},
    "agglomerative": {"threshold", "linkage", "metric"},
    "HDBScan": {"min_cluster_size", "min_samples", "metric"},
}
_SIM_LINKAGES = ("average", "complete", "single")

# Matching pairs returned by one range search call in SLINK (about 1.3 GB at most).
_PAIR_BUDGET = 1 << 26


def _canonical_cluster_type(cluster_type: str) -> str:
    for name in CLUSTER_TYPES:
        if isinstance(cluster_type, str) and cluster_type.lower() == name.lower():
            return name
    raise ValueError(f"cluster_type must be one of {CLUSTER_TYPES}, got {cluster_type!r}")


def resolve_cluster_params(
    cluster_type: str = "SLINK",
    sim_threshold: Optional[float] = None,
    cluster_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Check the clustering options and fill in defaults.

    :param cluster_type: "SLINK", "agglomerative" or "HDBScan" (any capitalization).
    :param sim_threshold: cosine similarity at or above which rows belong together (SLINK, agglomerative).
    :param cluster_params: method options. ``min_cluster_size`` (SLINK, HDBScan), ``min_samples`` (HDBScan),
        ``linkage`` (agglomerative: "average", "complete" or "single"; "ward" only with ``threshold``) and
        ``metric``. The original names "min cluster size", "min samples" and "clustering linkage" also work.
        ``threshold`` is the original way to set the cutoff: a cosine *distance* (1 - similarity), or a
        euclidean distance with ``metric="euclidean"``. Use ``sim_threshold`` instead.
    :return: dict with ``cluster_type`` and the resolved options.
    """
    cluster_type = _canonical_cluster_type(cluster_type)
    params = {}
    for key, value in (cluster_params or {}).items():
        params[_PARAM_ALIASES.get(key, key)] = value
    unknown = set(params) - _ALLOWED_PARAMS[cluster_type]
    if unknown:
        raise ValueError(f"Unknown cluster_params {sorted(unknown)} for {cluster_type}. "
                         f"Allowed: {sorted(_ALLOWED_PARAMS[cluster_type])}.")
    if sim_threshold is not None:
        if not -1.0 <= float(sim_threshold) <= 1.0:
            raise ValueError(f"sim_threshold is a cosine similarity between -1 and 1, got {sim_threshold}")
        if "threshold" in params:
            raise ValueError("Pass either sim_threshold (a similarity) or cluster_params['threshold'] (a distance), not both.")
        if params.get("metric", "cosine") != "cosine":
            raise ValueError("sim_threshold is a cosine similarity; drop cluster_params['metric'].")

    if cluster_type == "HDBScan":
        if sim_threshold is not None:
            raise ValueError("HDBScan has no similarity threshold. Set cluster_params={'min_cluster_size': ..., "
                             "'min_samples': ...}, or use SLINK or agglomerative with sim_threshold.")
        if params.get("metric", "euclidean") not in ("euclidean", "cosine"):
            raise ValueError("HDBScan supports metric 'euclidean' or 'cosine' (the same on normalized embeddings).")
        out = {"cluster_type": cluster_type, "min_cluster_size": int(params.get("min_cluster_size", 5)),
               "min_samples": int(params.get("min_samples", 1))}
        if out["min_cluster_size"] < 2 or out["min_samples"] < 1:
            raise ValueError("HDBScan needs min_cluster_size >= 2 and min_samples >= 1.")
        return out

    if sim_threshold is None and "threshold" not in params:
        raise ValueError(
            f"{cluster_type} clustering needs a similarity threshold, e.g. sim_threshold=0.85: rows whose cosine "
            "similarity is at least the threshold are grouped. There is no good default; tune it on rows whose "
            "clusters you know, or by reading a few clusters at several thresholds."
        )

    if cluster_type == "SLINK":
        metric = params.get("metric", "cosine")
        if sim_threshold is None:
            # legacy: DBSCAN eps as a distance between normalized embeddings
            distance = float(params["threshold"])
            if metric == "cosine":
                sim_threshold = 1.0 - distance
            elif metric == "euclidean":
                sim_threshold = 1.0 - distance ** 2 / 2.0
            else:
                raise ValueError("SLINK supports metric 'cosine' or 'euclidean'.")
        out = {"cluster_type": cluster_type, "sim_threshold": float(sim_threshold),
               "min_cluster_size": int(params.get("min_cluster_size", 2))}
        if out["min_cluster_size"] < 1:
            raise ValueError("SLINK needs min_cluster_size >= 1.")
        return out

    # agglomerative
    if sim_threshold is not None:
        linkage = params.get("linkage", "average")
        if linkage not in _SIM_LINKAGES:
            raise ValueError(f"With sim_threshold, linkage must be one of {_SIM_LINKAGES}; ward needs "
                             "cluster_params={'threshold': <euclidean distance>, 'linkage': 'ward'}.")
        return {"cluster_type": cluster_type, "linkage": linkage, "metric": "cosine",
                "distance_threshold": 1.0 - float(sim_threshold)}
    metric = params.get("metric", "euclidean")
    # ward, the default linkage, only works with euclidean distances
    linkage = params.get("linkage", "ward" if metric == "euclidean" else "average")
    return {"cluster_type": cluster_type, "linkage": linkage, "metric": metric,
            "distance_threshold": float(params["threshold"])}


def _normalize(embeddings: Any) -> np.ndarray:
    if hasattr(embeddings, "detach"):  # torch tensor
        embeddings = embeddings.detach().cpu().numpy()
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be a 2-D array (rows x dimensions), got shape {embeddings.shape}")
    return embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)


def _find(parent: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Roots of ``nodes``; compresses their paths."""
    roots = parent[nodes]
    while True:
        up = parent[roots]
        if np.array_equal(up, roots):
            break
        roots = up
    parent[nodes] = roots
    return roots


def _union(parent: np.ndarray, a: np.ndarray, b: np.ndarray) -> None:
    """Join the sets of each pair (a[i], b[i]). Every root is the smallest index of its set."""
    while len(a):
        root_a, root_b = _find(parent, a), _find(parent, b)
        differ = root_a != root_b
        if not differ.any():
            return
        low = np.minimum(root_a[differ], root_b[differ])
        high = np.maximum(root_a[differ], root_b[differ])
        np.minimum.at(parent, high, low)
        a, b = low, high


def _neighbour_pairs(embeddings: np.ndarray, sim_threshold: float):
    """Yield (rows, cols) of all pairs i < j with similarity >= sim_threshold, a bounded chunk at a time."""
    from linktransformer.search import range_search

    n = len(embeddings)
    step = max(1, min(n, _PAIR_BUDGET // max(n, 1)))
    start = 0
    while start < n:
        stop = min(n, start + step)
        lims, _, cols = range_search(embeddings[start:stop], embeddings, sim_threshold)
        rows = np.repeat(np.arange(start, stop), np.diff(lims))
        keep = rows < cols
        yield rows[keep], cols[keep]
        # grow the query chunk while matches are sparse, shrink it when they are dense
        if lims[-1] * 4 < _PAIR_BUDGET:
            step = min(n, step * 4)
        elif lims[-1] > _PAIR_BUDGET:
            step = max(1, step // 2)
        start = stop


def _slink(embeddings: np.ndarray, sim_threshold: float, min_cluster_size: int, weights: np.ndarray) -> np.ndarray:
    """
    DBSCAN with ``eps = 1 - sim_threshold`` on cosine distance and ``min_samples = min_cluster_size``, the
    weight of each point counting as that many copies of it. With min_cluster_size <= 2 this is single
    linkage. Returns labels with -1 for points in no cluster, numbered like scikit-learn's DBSCAN.
    """
    n = len(embeddings)
    parent = np.arange(n)
    if min_cluster_size <= 2:
        # every point with a neighbour (or a second copy) is a core point, so clusters are the linked groups
        core = weights >= min_cluster_size
        for rows, cols in _neighbour_pairs(embeddings, sim_threshold):
            _union(parent, rows, cols)
            core[rows] = True
            core[cols] = True
        border_rows = border_cols = np.empty(0, dtype=np.int64)
    else:
        mass = weights.astype(np.float64).copy()
        for rows, cols in _neighbour_pairs(embeddings, sim_threshold):
            mass += np.bincount(rows, weights=weights[cols], minlength=n) + np.bincount(cols, weights=weights[rows], minlength=n)
        core = mass >= min_cluster_size
        border_rows, border_cols = [], []
        for rows, cols in _neighbour_pairs(embeddings, sim_threshold):
            both = core[rows] & core[cols]
            _union(parent, rows[both], cols[both])
            for border, other in ((rows, cols), (cols, rows)):  # non-core points next to a core point
                edge = ~core[border] & core[other]
                border_rows.append(border[edge]); border_cols.append(other[edge])
        border_rows, border_cols = np.concatenate(border_rows), np.concatenate(border_cols)
    roots = _find(parent, np.arange(n))
    labels = np.where(core, roots, -1)
    if len(border_rows):
        # like scikit-learn: a border point joins the cluster with the smallest seed (its root) among its core neighbours
        best = np.full(n, n, dtype=np.int64)
        np.minimum.at(best, border_rows, roots[border_cols])
        labels = np.where(~core & (best < n), best, labels)
    # number clusters by their smallest point, as DBSCAN does
    clustered = labels >= 0
    _, labels[clustered] = np.unique(labels[clustered], return_inverse=True)
    return labels


def _physical_memory() -> Optional[int]:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):  # e.g. Windows
        return None


def _agglomerative(embeddings: np.ndarray, linkage: str, metric: str, distance_threshold: float) -> np.ndarray:
    n = len(embeddings)
    if n < 2:
        return np.zeros(n, dtype=np.int64)
    needed = n * (n - 1) // 2 * 8 * 2  # pairwise distances in float64, plus the linkage tree's working copy
    memory = _physical_memory()
    if memory is not None and needed > 0.8 * memory:
        raise MemoryError(
            f"Agglomerative clustering of {n:,} rows needs about {needed / 1e9:.0f} GB, more than this machine has "
            f"({memory / 1e9:.0f} GB). Use cluster_type='SLINK' (memory grows with the number of matching pairs), "
            "or split the rows with blocking_vars."
        )
    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, linkage=linkage, metric=metric)
    return model.fit(embeddings).labels_


def _hdbscan(embeddings: np.ndarray, min_cluster_size: int, min_samples: int) -> np.ndarray:
    if len(embeddings) <= max(min_cluster_size, min_samples):
        return np.full(len(embeddings), -1, dtype=np.int64)
    from sklearn.cluster import HDBSCAN

    # euclidean distance between normalized embeddings orders pairs exactly like cosine distance
    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean", copy=True)
    return model.fit(embeddings).labels_


def _cluster_points(embeddings: np.ndarray, options: Dict[str, Any], weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Labels for normalized embeddings, -1 for points in no cluster. ``weights`` (SLINK only) counts copies."""
    if len(embeddings) == 0:
        return np.empty(0, dtype=np.int64)
    kind = options["cluster_type"]
    if kind == "agglomerative" and options["linkage"] == "single" and options["metric"] in ("cosine", "euclidean"):
        # single linkage is SLINK, which scales; convert the distance cutoff to a similarity
        d = options["distance_threshold"]
        kind, options = "SLINK", {"sim_threshold": 1.0 - d if options["metric"] == "cosine" else 1.0 - d * d / 2.0,
                                  "min_cluster_size": 1}
    if kind == "SLINK":
        if weights is None:
            weights = np.ones(len(embeddings), dtype=np.int64)
        return _slink(embeddings, options["sim_threshold"], options["min_cluster_size"], np.asarray(weights))
    if weights is not None and np.any(np.asarray(weights) != 1):
        raise ValueError(f"{kind} clustering does not support weights")
    if kind == "agglomerative":
        return _agglomerative(embeddings, options["linkage"], options["metric"], options["distance_threshold"])
    return _hdbscan(embeddings, options["min_cluster_size"], options["min_samples"])


def own_ids_for_unclustered(labels: np.ndarray) -> np.ndarray:
    """Give every row labelled -1 a cluster of its own, and number clusters 0, 1, ... in order of first appearance."""
    labels = np.asarray(labels, dtype=np.int64).copy()
    noise = labels < 0
    labels[noise] = labels.max(initial=-1) + 1 + np.arange(noise.sum())
    _, first = np.unique(labels, return_index=True)
    order = np.empty(len(first), dtype=np.int64)
    order[np.argsort(first)] = np.arange(len(first))
    return order[np.unique(labels, return_inverse=True)[1]]


def cluster_embeddings(
    embeddings: Any,
    sim_threshold: Optional[float] = None,
    cluster_type: str = "SLINK",
    cluster_params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Cluster rows of an embedding matrix, for example embeddings from ``lt.infer_embeddings`` or another model.

    :param embeddings: (rows, dimensions) array or torch tensor. Rows are L2-normalized here.
    :param sim_threshold: cosine similarity at or above which rows belong together. Required for "SLINK" and
        "agglomerative"; not used by "HDBScan".
    :param cluster_type: "SLINK" (default), "agglomerative" or "HDBScan"; see :func:`resolve_cluster_params`.
    :param cluster_params: method options; see :func:`resolve_cluster_params`.
    :return: int64 array of cluster ids, one per row, numbered 0, 1, ... in order of first appearance. Rows with
        no near neighbour get a cluster of their own.
    """
    options = resolve_cluster_params(cluster_type, sim_threshold, cluster_params)
    return own_ids_for_unclustered(_cluster_points(_normalize(embeddings), options))


def cluster(cluster_type, cluster_params, corpus_embeddings, corpus_ids=None):
    """
    Original clustering entry point, kept for compatibility: prefer :func:`cluster_embeddings`.

    Returns scikit-learn style labels, with -1 for rows that SLINK or HDBScan leave unclustered.
    ``cluster_params["threshold"]`` is a distance here; ``corpus_ids`` is ignored.
    """
    options = resolve_cluster_params(cluster_type, None, cluster_params)
    return _cluster_points(_normalize(corpus_embeddings), options)


def clusters_from_edges(edges_list):
    """Identify clusters of passages given a dictionary of edges"""

    # clusters via NetworkX
    import networkx as nx

    G = nx.Graph()
    G.add_edges_from(edges_list)
    sub_graphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    sub_graph_dict = {}
    for i in range(len(sub_graphs)):
        sub_graph_dict[i] = list(sub_graphs[i].nodes())

    return sub_graph_dict


def edges_from_clusters(cluster_dict):
    """
    Convert every pair in a cluster into an edge
    """
    cluster_edges = []
    for cluster_id in list(cluster_dict.keys()):
        art_ids_list = cluster_dict[cluster_id]
        edge_list = [list(comb) for comb in combinations(art_ids_list, 2)]
        cluster_edges.extend(edge_list)

    return cluster_edges
