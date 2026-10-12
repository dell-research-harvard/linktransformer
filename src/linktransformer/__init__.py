"""Top-level package for LinkTransformer.

This module keeps the historical `import linktransformer as lt; lt.merge(...)` style
working while avoiding importing the full training stack at import time.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

__version__ = "0.1.18"

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


# Lazily expose the historically-exported API without eager star imports.
# Use explicit symbol->module mapping to avoid module/function name collisions.
_LAZY_EXPORTS = {
	# Inference APIs
	"merge": "linktransformer.infer",
	"merge_blocking": "linktransformer.infer",
	"aggregate_rows": "linktransformer.infer",
	"cluster_rows": "linktransformer.infer",
	"dedup_rows": "linktransformer.infer",
	"evaluate_pairs": "linktransformer.infer",
	"all_pair_combos_evaluate": "linktransformer.infer",
	"merge_knn": "linktransformer.infer",
	"merge_range": "linktransformer.infer",
	"classify_rows": "linktransformer.infer",
	"transform_rows": "linktransformer.infer",
	"merge_k_judge": "linktransformer.infer",
	"judge_merge_k_out": "linktransformer.infer",
	# Training APIs
	"train_model": "linktransformer.train_model",
	"create_new_train_config": "linktransformer.train_model",
	"train_clf_model": "linktransformer.train_clf_model",
	# Utilities commonly used via top-level import
	"load_model": "linktransformer.utils",
	"load_clf": "linktransformer.utils",
	"infer_embeddings": "linktransformer.utils",
	"serialize_columns": "linktransformer.utils",
	# Model classes
	"LinkTransformer": "linktransformer.modelling.LinkTransformer",
	"LinkTransformerClassifier": "linktransformer.modelling.LinkTransformerClassifier",
}


def __getattr__(name: str) -> Any:
	module_name = _LAZY_EXPORTS.get(name)
	if module_name is not None:
		module = importlib.import_module(module_name)
		value = getattr(module, name)
		globals()[name] = value
		return value
	raise AttributeError(f"module 'linktransformer' has no attribute {name!r}")


def __dir__() -> list[str]:
	return sorted(set(globals().keys()) | set(_LAZY_EXPORTS.keys()))


__all__ = [
	"__version__",
	"__MODEL_HUB_ORGANIZATION__",
	"DATA_DIR_PATH",
	"citation",
	"print_citation",
	*sorted(_LAZY_EXPORTS.keys()),
]




