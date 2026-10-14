import os
import subprocess
import sys
import types

import linktransformer as lt

# Every public name available as `lt.<name>` in 0.1.18 (these must keep resolving), plus judge_merge_k_out.
PUBLIC_API = [
    "DATA_DIR_PATH", "LinkTransformer", "LinkTransformerClassifier", "aggregate_rows", "all_pair_combos_evaluate",
    "apply_in_chunks", "check_and_prep_data", "check_clust_data", "classify_rows", "cluster", "cluster_rows",
    "clusters_from_edges", "convert_to_text", "cosine_similarity_corresponding_pairs",
    "cosine_similarity_corresponding_pairs_torch", "create_new_train_config", "dedup_rows", "evaluate_pairs",
    "get_completion_from_messages", "infer_embeddings", "infer_embeddings_with_gemini", "judge_merge_k_out", "load_clf",
    "load_model", "match_report",
    "merge", "merge_blocking", "merge_k_judge", "merge_knn", "merge_knn_openai", "merge_knn_with_llm", "merge_range",
    "openai_transform", "predict_rows_with_openai", "prep_clus_data", "prep_linkage_data", "prep_paired_label_data",
    "preprocess_any_data", "preprocess_mexican_tarrif_data", "serialize_columns", "tokenize_data_for_inference",
    "train_biencoder", "train_clf_model", "train_model", "transform_column", "transform_rows",
]

# Optional or training-only libraries that a plain inference workflow must not import.
NOT_NEEDED_FOR_INFERENCE = ["faiss", "wandb", "google.genai", "google.generativeai", "hdbscan", "evaluate"]


def test_lt_public_names_resolve():
    for name in PUBLIC_API:
        assert hasattr(lt, name), name
    assert sorted(lt.__all__) == sorted(PUBLIC_API + ["citation", "print_citation"])
    assert callable(lt.citation) and "Arora" in lt.citation()


def test_lt_functions_shadowing_submodules_stay_functions():
    # importing the submodules first must not replace the public functions of the same name
    import linktransformer.train_clf_model  # noqa: F401
    import linktransformer.train_model  # noqa: F401

    assert isinstance(lt.train_model, types.FunctionType)
    assert lt.train_model.__module__ == "linktransformer.train_model"
    assert isinstance(lt.train_clf_model, types.FunctionType)


def _modules_loaded_after(code):
    out = subprocess.run(
        [sys.executable, "-c", f"import sys\n{code}\nprint(','.join(sorted(sys.modules)))"],
        check=True, capture_output=True, text=True, env={**os.environ, "LINKTRANSFORMER_SILENCE_CITATION": "1"},
    ).stdout.strip().splitlines()[-1]
    return set(out.split(","))


def test_lt_import_is_lightweight():
    loaded = _modules_loaded_after("import linktransformer")
    heavy = {"torch", "transformers", "sentence_transformers", "sklearn", "pandas", *NOT_NEEDED_FOR_INFERENCE}
    assert not heavy & loaded, heavy & loaded


def test_lt_inference_does_not_import_optional_libraries():
    loaded = _modules_loaded_after("import linktransformer as lt\nlt.merge; lt.merge_knn; lt.cluster_rows")
    assert not set(NOT_NEEDED_FOR_INFERENCE) & loaded, set(NOT_NEEDED_FOR_INFERENCE) & loaded


def test_lt_match_report_is_lightweight():
    # checking a merge result should not pull in torch or the model libraries
    loaded = _modules_loaded_after("import linktransformer as lt\nlt.match_report")
    assert not {"torch", "transformers", "sentence_transformers", "openai"} & loaded
