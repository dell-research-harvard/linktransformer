# LinkTransformer

[![arXiv](https://img.shields.io/badge/arXiv-2309.00789-b31b1b.svg)](https://arxiv.org/abs/2309.00789)
![LinkTransformer demo](https://github.com/dell-research-harvard/linktransformer/assets/60428420/15162828-b0fb-4ee3-8a0f-fdf3371d10a0)

LinkTransformer is a Python package for semantic record linkage, candidate retrieval, row transformation, clustering, and text classification over tabular data.

- Paper: https://aclanthology.org/2024.acl-demos.21/ (ACL 2024 System Demonstrations; preprint: https://arxiv.org/abs/2309.00789)
- Website: https://linktransformer.github.io/
- Demo video: https://www.youtube.com/watch?v=Sn47nmCvV9M

## Tutorials

- Link records with LinkTransformer: https://colab.research.google.com/drive/1OqUB8sqpUvrnC8oa_1RoOUzV6DaAKL4N?usp=sharing
- Train your own LinkTransformer model: https://colab.research.google.com/drive/1tHitPGjMMI2Nvh4wwA8rdcbYfbLaJDvg?usp=sharing
- Classify text with LinkTransformer: https://colab.research.google.com/drive/1hSh_p8j7LP2RfdtxrPslOfnogC_CbYw5?usp=sharing
- Demo app (Hugging Face Space): https://huggingface.co/spaces/96abhishekarora/linktransformer_merge
- Feature deck: https://www.dropbox.com/scl/fi/dquxru8bndlyf9na14cw6/A-python-package-to-do-easy-record-linkage-using-Transformer-models.pdf?rlkey=fiv7j6c0vgl901y940054eptk&dl=0

More tutorials are coming soon.

## Installation

```bash
pip install linktransformer
```

The base install covers linking, clustering, deduplication, classification, and OpenAI-based features. Optional extras add:

| Extra | Adds | Install |
|---|---|---|
| `train` | Fine-tuning with `train_model` and `train_clf_model` | `pip install "linktransformer[train]"` |
| `wandb` | Weights & Biases logging during training | `pip install "linktransformer[wandb]"` |
| `gemini` | Gemini embeddings and Gemini as the `merge_k_judge` LLM | `pip install "linktransformer[gemini]"` |
| `all` | Everything above | `pip install "linktransformer[all]"` |

On Apple Silicon Macs, use a native (arm64) Python. An Intel Python running under Rosetta, such as an x86 Miniconda, cannot install current versions of PyTorch.

## Quick Start

```python
import os
import pandas as pd
import linktransformer as lt

left_df = pd.DataFrame({"CompanyName": ["Tech Corporation"], "Country": ["USA"]})
right_df = pd.DataFrame({"CompanyName": ["Tech Corp"], "Country": ["USA"]})

out = lt.merge(
    left_df,
    right_df,
    on=["CompanyName", "Country"],
    model="sentence-transformers/all-MiniLM-L6-v2",
)
print(out[["CompanyName_x", "CompanyName_y", "score"]])
```

## NEW RELEASE: End-to-end linkage Workflow: `merge_k_judge` (End-to-End Record Linkage)

`merge_k_judge` is the recommended end-to-end linkage API when you want both retrieval and LLM adjudication with confidence.

1. Retrieve top-`k` candidates with embeddings (`merge_knn`)
2. Judge each candidate pair with an LLM
3. Return match decisions and confidence scores

```python
judged = lt.merge_k_judge(
    df1=left_df,
    df2=right_df,
    on=["CompanyName", "Country"],
    k=5,
    knn_sbert_model="sentence-transformers/all-MiniLM-L6-v2",
    judge_llm_model="gpt-4o-mini",
    llm_provider="openai",
    openai_key=os.getenv("OPENAI_API_KEY"),
)

# key output columns:
# - score (retrieval similarity)
# - is_match (bool)
# - confidence (float in [0, 1] when available)
```

You can also combine providers (for example OpenAI embeddings retrieval + Gemini judge) by setting `knn_api_model`, `judge_llm_model`, and `llm_provider` explicitly.

## Core APIs

### 1) Link two dataframes

- `lt.merge(...)`: semantic 1:1 / 1:m / m:1 linkage.
- `lt.merge_knn(...)`: top-`k` candidate retrieval.
- `lt.merge_blocking(...)`: run merge within blocks to do fuzzy merge within exact matches.
- `lt.aggregate_rows(...)`: map fine rows to coarser labels.

```python
matches = lt.merge_knn(
    left_df,
    right_df,
    on=["CompanyName", "Country"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    k=3,
)
```

### 2) Transform rows with LLM prompts

Use `lt.transform_rows(...)` to normalize, rewrite, or standardize values in one or more columns. Eg : Fix OCR errors in the Column, Standardize names. 

```python
cleaned = lt.transform_rows(
    left_df,
    on=["CompanyName", "Country"],
    model="gpt-4o-mini",
    openai_key=os.getenv("OPENAI_API_KEY"),
    openai_prompt=(
        "Standardize organization names and country strings for record linkage. "
        "Return a JSON list in the same order."
    ),
)
# adds: transformed_CompanyName-Country
```

### 3) Cluster and deduplicate

- `lt.cluster_rows(...)`: cluster semantically similar rows.
- `lt.dedup_rows(...)`: cluster + keep representative rows.

```python
deduped = lt.dedup_rows(
    left_df,
    on="CompanyName",
    model="sentence-transformers/all-MiniLM-L6-v2",
    cluster_type="agglomerative",
    cluster_params={"threshold": 0.7},
)
```

### 4) Evaluate matched pairs

- `lt.evaluate_pairs(...)`: similarity over known pairs.
- `lt.all_pair_combos_evaluate(...)`: dense pairwise scoring.

### 5) Classification

- `lt.classify_rows(...)`: classify rows with HF or OpenAI chat models.
- `lt.train_clf_model(...)`: train a custom row classifier.

### 6) Train linkage models

- `lt.train_model(...)`: train a linkage model from paired or clustered data.

## Caching embeddings

Pass a folder as `embedding_cache` to any function that embeds text (`merge`, `merge_knn`, `merge_range`, `merge_k_judge`, `merge_blocking`, `aggregate_rows`, `evaluate_pairs`, `all_pair_combos_evaluate`, `cluster_rows`, `dedup_rows`, `infer_embeddings`). Embeddings are stored there per model and text, so later calls embed only texts that are not cached yet. For example, embed a large reference table once and then match small batches of new rows against it:

```python
out = lt.merge_knn(new_rows, reference_df, on="CompanyName", model="dell-research-harvard/lt-wikidata-comp-en",
                   k=5, embedding_cache="embeddings_cache")
```

Both sides and all functions share one cache. Local models are identified by a fingerprint of their weights, so retraining a model never returns stale embeddings; OpenAI and Gemini models are identified by name.

## Long-running API jobs

API-based workflows (OpenAI/Gemini embeddings, `merge_k_judge`, `judge_merge_k_out`, `classify_rows`, `transform_rows`) accept `checkpoint_path` (a JSONL progress file), `resume=True` to continue an interrupted run, `failure_log_path` to record failed calls, and `retry_policy={"max_retries": 5, "base_sleep_seconds": 15, "jitter_seconds": 2}`. `judge_workers` / `api_workers` run API calls in parallel.

## Checking results and API cost

These helpers return small dicts of plain Python values, which suit scripts and coding agents:

```python
# how many LLM calls (and roughly how many input tokens) a run would make, without calling the LLM
plan = lt.merge_k_judge(left_df, right_df, on="CompanyName", k=5, knn_sbert_model="all-MiniLM-L6-v2",
                        judge_llm_model="gpt-4o-mini", dry_run=True)
print(plan["llm_calls"], plan["estimated_input_tokens"])

# summary of a merge: unmatched rows, score quantiles, one-to-many links, samples near a threshold
matches = lt.merge_knn(left_df, right_df, on="CompanyName", k=3)
report = lt.match_report(matches, left_df, right_df, on="CompanyName", threshold=0.8)
```

`dry_run=True` also works on `judge_merge_k_out`, `classify_rows` and `transform_rows`. It applies checkpoint resume, already-judged rows, score-based skipping and batching as the real run would. `merge_k_judge` only embeds the data during a dry run when `drop_sim_threshold` or a `judge_skip_*_thresh` makes the number of calls depend on similarity scores. Token counts are rough estimates: about 4 characters per token, with more tokens for non-English text. Without a `threshold`, `match_report` suggests one: the cutoff that agrees best with the LLM's judgments, if there are any, or else a split of the score histogram. Check the returned samples before relying on it.

Set `LINKTRANSFORMER_PROGRESS=0` to turn off every progress bar, including transformers' weight-loading bars. Functions that show a bar also accept `progress_bar=False`.

## Provider Notes

- Japanese models built on mLUKE (`lt-wikidata-comp-ja`, `lt-wikidata-comp-prod-ind-ja`, `lt-historicjapan-*`) need transformers 4.x: `pip install "transformers>=4.56,<5" "sentence-transformers>=5,<6"`. transformers 5 does not tokenize them correctly, and LinkTransformer raises an error rather than returning wrong embeddings.

- OpenAI key: set `OPENAI_API_KEY` or pass `openai_key`.
- Gemini key: set `GEMINI_API_KEY` or pass `gemini_key`.
- API embedding models and local SBERT models are both supported.
- For multi-column API retrieval, LinkTransformer serializes columns safely using `<SEP>`.

## Development and testing

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
pytest -m "not slow"   # fast suite, a few minutes on a laptop CPU
pytest -m slow         # full training tests; long on CPU
```

To check the oldest supported dependency versions, install with `uv pip install --resolution lowest-direct -e ".[dev]"`. CI runs both on Linux and Apple Silicon macOS for Python 3.11-3.13. On a Mac, use a native arm64 Python (uv's managed Pythons are); tests that call the OpenAI or Gemini APIs run only when `OPENAI_API_KEY` / `GEMINI_API_KEY` are set.

Tests use `test_lt_*` naming to mirror the package API surface and make workflows discoverable.

## Contributing

Issues and pull requests are welcome.

## Citation

If you use LinkTransformer in your research, please cite:

```bibtex
@inproceedings{arora-dell-2024-linktransformer,
    title = "{L}ink{T}ransformer: A Unified Package for Record Linkage with Transformer Language Models",
    author = "Arora, Abhishek and Dell, Melissa",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)",
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-demos.21/",
    doi = "10.18653/v1/2024.acl-demos.21",
    pages = "221--231"
}
```

## License

This project is licensed under the MIT License. See `LICENSE`.

## Maintainers

- Sam Jones (`samuelcaronnajones`)
- Abhishek Arora (`econabhishek`)
- Yiyang Chen (`oooyiyangc`)
