# __init__.py
#
# Public names are imported lazily on first access (PEP 562), so `import linktransformer`
# stays fast and only the modules a workflow actually uses (inference, training, API
# providers) get imported.

import importlib
import os
import sys
import types
from typing import TYPE_CHECKING

__version__ = "0.2.0"

# For compatibility with sentence-transformers / existing model-card behavior.
from .constants import __MODEL_HUB_ORGANIZATION__
from .data import DATA_DIR_PATH
from .citation import citation, print_citation

_CITATION_PRINTED = False


def _maybe_print_citation() -> None:
    global _CITATION_PRINTED
    if _CITATION_PRINTED:
        return
    if os.getenv("LINKTRANSFORMER_SILENCE_CITATION", "").strip() in {"1", "true", "True", "yes", "YES"}:
        _CITATION_PRINTED = True
        return

    print(citation())
    _CITATION_PRINTED = True


_maybe_print_citation()

_LAZY_ATTRS = {
    "linktransformer.infer": [
        "aggregate_rows", "all_pair_combos_evaluate", "classify_rows", "cluster_rows", "dedup_rows",
        "evaluate_pairs", "judge_merge_k_out", "merge", "merge_blocking", "merge_k_judge", "merge_knn", "merge_knn_openai",
        "merge_knn_with_llm", "merge_range", "transform_rows",
    ],
    "linktransformer.utils": [
        "apply_in_chunks", "cosine_similarity_corresponding_pairs", "cosine_similarity_corresponding_pairs_torch",
        "get_completion_from_messages", "infer_embeddings", "infer_embeddings_with_gemini", "load_clf",
        "load_model", "openai_transform", "predict_rows_with_openai", "serialize_columns",
        "tokenize_data_for_inference", "transform_column",
    ],
    "linktransformer.cluster_fns": ["cluster", "clusters_from_edges"],
    "linktransformer.diagnostics": ["match_report"],
    "linktransformer.preprocess": [
        "check_and_prep_data", "check_clust_data", "convert_to_text", "prep_clus_data", "prep_linkage_data",
        "prep_paired_label_data", "preprocess_any_data", "preprocess_mexican_tarrif_data",
    ],
    "linktransformer.train_model": ["create_new_train_config", "train_model"],
    "linktransformer.train_clf_model": ["train_clf_model"],
    "linktransformer.modified_sbert.train": ["train_biencoder"],
    "linktransformer.modelling.LinkTransformer": ["LinkTransformer"],
    "linktransformer.modelling.LinkTransformerClassifier": ["LinkTransformerClassifier"],
}
_ATTR_TO_MODULE = {attr: module for module, attrs in _LAZY_ATTRS.items() for attr in attrs}
_SUBMODULES = {
    "api_runtime", "cluster_fns", "configs", "constants", "data", "diagnostics", "embedding_cache", "infer",
    "modelling", "modified_sbert", "preprocess", "progress", "search", "train_clf_model", "train_model", "utils",
}

__all__ = ["DATA_DIR_PATH", "citation", "print_citation", *sorted(_ATTR_TO_MODULE)]


def __getattr__(name):
    if name in _ATTR_TO_MODULE:
        value = getattr(importlib.import_module(_ATTR_TO_MODULE[name]), name)
    elif name in _SUBMODULES:
        value = importlib.import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value  # cache so later lookups skip __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | _SUBMODULES)


# `train_model`, `train_clf_model` and `citation` are both public functions and submodules. Importing a
# submodule makes Python bind it as an attribute of this package, which would shadow the
# function, so the package ignores that rebinding for these names.
_FUNCTIONS_NAMED_LIKE_SUBMODULES = {"train_model", "train_clf_model", "citation"}


class _LinkTransformerPackage(types.ModuleType):
    def __setattr__(self, name, value):
        if name in _FUNCTIONS_NAMED_LIKE_SUBMODULES and isinstance(value, types.ModuleType):
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _LinkTransformerPackage


if TYPE_CHECKING:  # let static type checkers and IDEs see the real definitions
    from .cluster_fns import cluster, clusters_from_edges
    from .diagnostics import match_report
    from .infer import (
        aggregate_rows, all_pair_combos_evaluate, classify_rows, cluster_rows, dedup_rows, evaluate_pairs,
        judge_merge_k_out, merge, merge_blocking, merge_k_judge, merge_knn, merge_knn_openai, merge_knn_with_llm,
        merge_range, transform_rows,
    )
    from .modelling.LinkTransformer import LinkTransformer
    from .modelling.LinkTransformerClassifier import LinkTransformerClassifier
    from .modified_sbert.train import train_biencoder
    from .preprocess import (
        check_and_prep_data, check_clust_data, convert_to_text, prep_clus_data, prep_linkage_data,
        prep_paired_label_data, preprocess_any_data, preprocess_mexican_tarrif_data,
    )
    from .train_clf_model import train_clf_model
    from .train_model import create_new_train_config, train_model
    from .utils import (
        apply_in_chunks, cosine_similarity_corresponding_pairs, cosine_similarity_corresponding_pairs_torch,
        get_completion_from_messages, infer_embeddings, infer_embeddings_with_gemini, load_clf, load_model,
        openai_transform, predict_rows_with_openai, serialize_columns, tokenize_data_for_inference,
        transform_column,
    )
