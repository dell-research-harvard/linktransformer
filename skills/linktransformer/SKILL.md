---
name: linktransformer
description: Record linkage, fuzzy merges, entity resolution, deduplication and clustering of messy names or records (firms, people, products, places, OCR text) in pandas DataFrames with the linktransformer Python package (`import linktransformer as lt`). Use when joining tables that lack a clean shared key, matching names across spellings or languages, retrieving top-k candidate matches, verifying matches with an LLM judge, deduplicating or clustering rows, mapping fine categories to coarse ones, classifying or cleaning rows with an LLM, or fine-tuning a linkage model. Covers choosing the function and model, validating a similarity threshold, LLM cost guardrails and long-running jobs.
---

# LinkTransformer

LinkTransformer embeds the key column(s) of each row with a transformer model and links rows by cosine similarity. Everything is a function on pandas DataFrames: call the Python API directly. This skill describes version 0.2.0; check it with `python -c "import linktransformer as lt; print(lt.__version__)"`.

```bash
pip install linktransformer              # extras: [train] fine-tuning, [gemini] Gemini models, [all]
export LINKTRANSFORMER_SILENCE_CITATION=1  # otherwise every import prints a BibTeX citation
```

## Pick the function

| Task | Function | Notes |
|---|---|---|
| One best match in `df2` for every row of `df1` | `lt.merge` | Nearest neighbour, m:1. Every left row gets exactly one right row, with no threshold. Filter on `score` yourself. |
| Top-k candidates per left row | `lt.merge_knn(k=...)` | `len(df1) * min(k, len(df2))` rows. `drop_sim_threshold=` drops pairs below a score. Use it for review, for 1:m links, and as LLM-judge input. |
| All matches above a similarity | `lt.merge_range(sim_threshold=...)` | Variable number of matches per left row. Left rows with no match are kept with NaN `score`. |
| Fuzzy match only within exact groups (same country, year, ...) | `lt.merge_blocking(blocking_vars=[...])` | Runs `merge` inside each block. Rows in blocks missing from the other side come back with `match_status` `left_only_block` / `right_only_block`. |
| Retrieve candidates, then have an LLM decide | `lt.merge_k_judge` | `merge_knn` + `judge_merge_k_out`. Costs one LLM call per judged pair. See [references/llm-workflows.md](references/llm-workflows.md). |
| Judge a candidate table you already have, resumably | `lt.judge_merge_k_out` | Skips rows that already have `llm_*` values. Prefer it for large runs. |
| Map fine labels to a coarse reference list | `lt.aggregate_rows(df, ref_df, left_on=, right_on=)` | Same as `merge` with `ref_df` on the right. |
| Group near-duplicates | `lt.cluster_rows` | Adds a `cluster` column; `-1` means unclustered for SLINK/HDBScan. |
| Drop near-duplicates | `lt.dedup_rows` | Keeps the first row per cluster. Read the dedup gotcha below first. |
| Score known pairs (two columns of one df) | `lt.evaluate_pairs` | Adds `score`; use it to pick a threshold from labeled pairs. |
| Classify rows | `lt.classify_rows` | Local HF classifier, or an OpenAI/Gemini chat model in batches. |
| Clean or standardize text before linking | `lt.transform_rows` | LLM rewrite in batches (OCR fixes, name normalization). API only. |
| Embeddings only | `lt.infer_embeddings(strings, model)` | Returns a numpy array. Normalize it before taking dot products. |
| Fine-tune on your own matched pairs or clusters | `lt.train_model`, `lt.train_clf_model` | Needs the `[train]` extra. See [references/recipes.md](references/recipes.md#train-a-linkage-model). |

`merge_knn_with_llm` and `merge_knn_openai` are backward-compatible aliases of `merge_k_judge`; don't use them in new code.

## Minimal example

```python
import os
import pandas as pd
import linktransformer as lt

left = pd.read_csv(os.path.join(lt.DATA_DIR_PATH, "toy_comp_1.csv"))   # CompanyName, Industry, Country, ...
right = pd.read_csv(os.path.join(lt.DATA_DIR_PATH, "toy_comp_2.csv"))  # CompanyName, Revenue, Country, ...

out = lt.merge_knn(left, right, on="CompanyName", model="dell-research-harvard/lt-wikidata-comp-en",
                   k=3, embedding_cache="lt_cache")
print(out[["CompanyName_x", "CompanyName_y", "score"]].head(6))
```

Output: the left columns, then the right columns, then `score` (cosine similarity, higher is closer). Column names present on both sides get the `_x` / `_y` suffixes (`suffixes=` changes them). `id_lt_x` / `id_lt_y` are the 0-based row positions in `left` / `right`, so you can join results back to the inputs.

## Choose a model

- **Default:** `model="all-MiniLM-L6-v2"` (sentence-transformers). It is small and fast, a reasonable baseline for English names.
- **LinkTransformer models on the Hugging Face Hub** (`dell-research-harvard/...`), trained for linkage:

| Domain | Models |
|---|---|
| Company names (Wikidata aliases) | `lt-wikidata-comp-en`, `-es`, `-fr`, `-de`, `-zh`, `-ja`, and `lt-wikidata-comp-multi` (multilingual and cross-lingual) |
| Modern Japanese companies (company, product, industry) | `lt-wikidata-comp-prod-ind-ja` |
| Historic Japanese firms | `lt-historicjapan-supcon`, `lt-historicjapan-onlinecontrastive` |
| Product descriptions (UN product classifications) | `lt-un-data-fine-fine-{en,es,fr,multi}` (product to product), `lt-un-data-fine-coarse-{...}` (product to coarse category), `lt-un-data-fine-industry-{...}` (product to ISIC industry) |
| Spanish product descriptions in Mexican trade data | `lt-mexicantrade4748` |

  Prefix each name with `dell-research-harvard/`. The Japanese models are built on mLUKE and need `pip install "transformers>=4.56,<5" "sentence-transformers>=5,<6"`; LinkTransformer raises an error under transformers 5.
- **Any other** sentence-transformers or Hugging Face model id, or a local folder, also works, including a model you trained with `lt.train_model`.
- **API embeddings:** OpenAI model names contain `text-embedding` (e.g. `"text-embedding-3-small"`); Gemini names contain `gemini` (e.g. `"gemini-embedding-001"`, needs the `[gemini]` extra). Keys come from the `OPENAI_API_KEY` / `GEMINI_API_KEY` environment variables. Never write a key into code, logs or notebooks. If a key is missing, ask the user to export it. Don't pass `openai_key=` / `gemini_key=` together with a *local* model name: `merge` and its siblings then send that name to the API and fail.

Which to start with: pick the LinkTransformer model for your domain and language if one exists, otherwise the default model. Compare candidates on the same labeled sample with `evaluate_pairs` before committing to one.

## Validate before trusting a threshold

Embedding scores have no universal cutoff: it depends on the model and the data. Run this loop on every new task and pass `embedding_cache=` so reruns only embed new text.

```python
MODEL = "dell-research-harvard/lt-wikidata-comp-en"
cands = lt.merge_knn(left, right, on="CompanyName", model=MODEL, k=1, embedding_cache="lt_cache")

# 1. Score distribution of each left row's best match
print(cands["score"].describe())
print(pd.cut(cands["score"], [-1, .5, .6, .7, .8, .9, 1.0]).value_counts().sort_index())

# 2. Read pairs near a candidate threshold, where the errors are
t = 0.75
near = cands[cands["score"].between(t - 0.05, t + 0.05)]
print(near.sample(min(20, len(near)), random_state=0)[["CompanyName_x", "CompanyName_y", "score"]])

# 3. Apply the threshold you settled on
links = lt.merge_knn(left, right, on="CompanyName", model=MODEL, k=1, drop_sim_threshold=t,
                     embedding_cache="lt_cache")
```

Show the user the near-threshold pairs and let them confirm the cutoff. `merge` has no threshold argument: filter `out[out["score"] >= t]` yourself.

With labeled pairs (one row per pair, with a 0/1 label), choose the threshold from data:

```python
import numpy as np

pairs = pd.read_csv(os.path.join(lt.DATA_DIR_PATH, "toy_pairs.csv"))  # company_name_1, company_name_2, label
ev = lt.evaluate_pairs(pairs, model=MODEL, left_on="company_name_1", right_on="company_name_2",
                       embedding_cache="lt_cache")
for t in np.arange(0.5, 1.0, 0.05):
    pred, y = ev["score"] >= t, ev["label"] == 1
    tp = (pred & y).sum()
    precision, recall = tp / max(pred.sum(), 1), tp / max(y.sum(), 1)
    print(f"t={t:.2f}  precision={precision:.2f}  recall={recall:.2f}")
```

`evaluate_pairs` scores each given pair. It doesn't check whether the true partner is the *nearest* neighbour, so also look at the `merge_knn` output with `k` > 1.

## LLM cost guardrail

`merge_k_judge`, `judge_merge_k_out`, `classify_rows` and `transform_rows` with an API model, plus API embeddings, cost money. Before any full run:

1. **Count the calls.** Judging makes one call per candidate pair with `judge_skip_lower_thresh < score < judge_skip_upper_thresh`. `classify_rows` makes `ceil(n_rows / batch_size)` calls (default `batch_size=128`). `transform_rows` makes `ceil(n_non_null_rows / batch_size)` calls (default 50). The formulas and a token estimate are in [references/llm-workflows.md](references/llm-workflows.md#estimate-cost-before-running).
2. **Pilot on a small sample** (e.g. `left.sample(30, random_state=0)`) and read `llm_raw_response` / the predictions before scaling up.
3. **Confirm with the user** before the full run when it needs more than about 1,000 LLM calls, or the estimate exceeds about US$5 or their stated budget. Report the call count, the model and the estimated cost. Look up current prices; don't assume them.
4. Narrow the judged set first: use a lower `k`, `drop_sim_threshold`, and skip thresholds so pairs that are clearly matches or clearly not skip the LLM.

## Long jobs

- For anything that runs more than a couple of minutes, write a script file and run it in the background (`nohup python link_job.py > link_job.log 2>&1 &`) or as a Slurm job. Don't use `python -c` or a notebook cell. Poll the log; don't wait on the process.
- Save intermediate outputs: the candidate table before judging, the judged table after. Then a crash costs only the unfinished stage.
- API functions take `checkpoint_path=` (a JSONL file), `resume=True`, `failure_log_path=` and `retry_policy={"max_retries": 5, "base_sleep_seconds": 15, "jitter_seconds": 2}`. Rerunning the same script with `resume=True` skips finished work. `judge_merge_k_out` also skips rows whose `llm_is_match`, `llm_confidence` and `llm_raw_response` are already filled. Checkpoints are keyed by row position, so resume against the same saved candidate file, never a recomputed one.
- `judge_workers=` (judge) and `api_workers=` (classify/transform) run API calls in parallel. Start at 4 and raise it if there are no rate-limit errors.
- Embedding and search run on CUDA automatically when a GPU is visible.

A complete script template, plus a Slurm variant, is in [references/llm-workflows.md](references/llm-workflows.md#long-job-template).

## Gotchas

- **`on` defaults to every column name the two frames share**, IDs included (toy_comp_1/2 share `CompanyID`, `CompanyName` and `Country`). Always pass `on=`, or `left_on=` / `right_on=`.
- **Multi-column keys are serialized into one string per row**, joined by the model tokenizer's separator token (`"TechCorp[SEP]USA"` for MiniLM). API embedding models use `"<SEP>"`; `transform_rows` and API `classify_rows` join with a space. NaN becomes `""`. Order matters, so list left and right columns in corresponding order, and pass `left_on` and `right_on` as the same type (both strings or both lists).
- **`merge_type` is deprecated.** It only warns when keys are not unique, so don't pass it. `merge` never leaves a left row unmatched and may reuse a right row many times.
- **An existing `id_lt` column in either input raises an error**; rename it first.
- **Cluster thresholds are distances, not similarities.** SLINK is DBSCAN with `eps=threshold` on cosine distance, so `threshold: 0.1` means similarity ≥ 0.9. With `cluster_params=None` the threshold is 0.5 (similarity ≥ 0.5), which is loose, and single linkage chains clusters. Always pass `cluster_params`.
- **`dedup_rows` with SLINK or HDBScan collapses all unclustered rows into one.** It keeps one row per `cluster` label, and every unclustered row has the label `-1`. Use `cluster_type="agglomerative"` with explicit params (it has no `-1`), or use `cluster_rows` and keep the `-1` rows yourself. See [references/recipes.md](references/recipes.md#deduplicate-and-cluster).
- **`agglomerative` without `cluster_params` raises an error** (ward linkage with cosine). Pass e.g. `{"threshold": 0.3, "clustering linkage": "average", "metric": "cosine"}`; the threshold is then a cosine distance.
- **The LLM judge's output columns are `llm_is_match` (0/1), `llm_confidence`, `llm_raw_response` and `llm_input_payload`**, not `is_match` / `confidence`. Rows skipped by the skip thresholds get the sentinel `99` in `llm_is_match` and `llm_confidence`. `confidence_threshold=` keeps those `99` rows, including those skipped for a *low* score. Filter on `score` and `llm_is_match` yourself instead.
- **A judge reply wrapped in a ```` ```json ```` fence is mis-parsed as a match** (0.2.0 bug). Check `llm_raw_response` in the pilot; [references/llm-workflows.md](references/llm-workflows.md#check-the-judge-output) has a re-parse snippet.
- **`classify_rows` and `transform_rows` use an API only when `openai_key=` or `gemini_key=` is passed explicitly**, e.g. `openai_key=os.environ["OPENAI_API_KEY"]`. Without a key, `classify_rows` treats `model` as a local classifier. Output columns are `clf_preds_<col>` and `transformed_<col>`; multiple columns are joined with `-`.
- **Progress bars:** judging, classification and transforms take `progress_bar=False`. The sentence-transformers "Batches" bar during embedding always prints to stderr in 0.2.0. `HF_HUB_DISABLE_PROGRESS_BARS=1` silences the model-loading bars. In background jobs, redirect stderr to the log.
- **`merge_k_judge` warns** about which retrieval model it resolved. Pass `knn_sbert_model=` (local) or `knn_api_model=` (API embeddings) explicitly; otherwise it retrieves with `model`, which defaults to MiniLM.
- **The embedding cache is keyed by model weights and the exact text.** Changing `on` or the column order changes the text, so those rows are re-embedded. One cache folder can serve every function and both sides of a merge.

## More

- [references/recipes.md](references/recipes.md): multi-column and cross-lingual links, blocking, `merge_range`, aggregation, dedup and clustering, local classification, training.
- [references/llm-workflows.md](references/llm-workflows.md): LLM judging end to end, cost estimation, output checks, `classify_rows` / `transform_rows` with APIs, the long-job script.
