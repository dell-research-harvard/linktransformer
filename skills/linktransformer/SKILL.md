---
name: linktransformer
description: Record linkage, fuzzy merges, entity resolution, deduplication and clustering of messy names or records (firms, people, products, places, OCR text) in pandas DataFrames with the linktransformer Python package (`import linktransformer as lt`). Use when joining tables that lack a clean shared key, matching names across spellings or languages, retrieving top-k candidate matches, verifying matches with an LLM judge, deduplicating or clustering rows, mapping fine categories to coarse ones, classifying or cleaning rows with an LLM, or fine-tuning a linkage model. Covers choosing the function and model, validating a similarity threshold, LLM cost guardrails and long-running jobs.
---

# LinkTransformer

LinkTransformer embeds the key column(s) of each row with a transformer model and links rows by cosine similarity. Everything is a function on pandas DataFrames: call the Python API directly. This skill describes version 0.2.0 with `lt.match_report` and `dry_run=`. If `python -c "import linktransformer as lt; print(lt.__version__, hasattr(lt, 'match_report'))"` prints an older version or `False`, upgrade the package.

```bash
pip install linktransformer              # extras: [train] fine-tuning, [gemini] Gemini models, [all]
export LINKTRANSFORMER_SILENCE_CITATION=1  # otherwise every import prints a BibTeX citation
export LINKTRANSFORMER_PROGRESS=0          # turn off every progress bar, so bars don't fill your context or logs
export TRANSFORMERS_VERBOSITY=error        # hide transformers' model-load reports, which are not progress bars
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
| Group near-duplicates | `lt.cluster_rows(sim_threshold=...)` | Adds `cluster` (an id for every row) and `cluster_size`; `cluster_size == 1` means no near-duplicate. `blocking_vars=` clusters within exact groups. |
| Drop near-duplicates | `lt.dedup_rows(sim_threshold=...)` | Keeps one row per cluster: the first, or the most central with `keep="medoid"`. |
| Cluster embeddings you already have | `lt.cluster_embeddings(emb, sim_threshold=...)` | Returns one cluster id per row. |
| Check a merge's output | `lt.match_report(out, df1, df2, on=...)` | Small JSON-ready dict: score quantiles, a suggested threshold, unmatched and one-to-many rows, LLM judge statistics, sample pairs near the threshold. Read it instead of printing the dataframe. |
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

Embedding scores have no universal cutoff: it depends on the model and the data. Run this loop on every new task, and pass `embedding_cache=` so reruns only embed new text. `lt.match_report` summarizes a merge in a small dict, so you never need to print the dataframe.

```python
import json

MODEL = "dell-research-harvard/lt-wikidata-comp-en"
cands = lt.merge_knn(left, right, on="CompanyName", model=MODEL, k=3, embedding_cache="lt_cache")

# 1. Score quantiles, a suggested threshold, and sample pairs on both sides of it
report = lt.match_report(cands, left, right, on="CompanyName")
print(json.dumps(report["score"]), json.dumps(report["threshold"]), report["notes"], sep="\n")
for s in report["samples"]:
    print(s["score"], s["side"], s["left"], "<->", s["right"])

# 2. Re-check at the cutoff you are considering, with more samples near it
t = 0.75
report = lt.match_report(cands, left, right, on="CompanyName", threshold=t, n_samples=10)
print(json.dumps(report["left"]["at_threshold"]), json.dumps(report["right"]["at_threshold"]))

# 3. Apply the threshold you settled on
links = cands[cands["score"] >= t]
```

Reading the report:

- `threshold` holds `value` and `source`. `source` is `"given"`, `"score_histogram"` (the Otsu split of the scores) or `"llm_judgments"` (the cutoff with the best F1 against `llm_is_match`). A suggested value is only a starting point.
- `samples` are the pairs nearest the threshold, half at or above it and half below. That's where the errors are. Show them to the user and let them confirm the cutoff.
- `left` / `right` count matched and `unmatched` rows (pass `df1` and `df2` for those). They also count rows linked to several rows of the other side (`with_multiple_right` / `with_multiple_left`, `max_right_per_key` / `max_left_per_key`), over all pairs and under `at_threshold` and `llm_matches`.
- `llm` (judge output only) has `rows_judged`, `rows_skipped_by_score`, `match_rate`, `confidence` quantiles and `disagreements_at_threshold`. `match_status` counts `merge_blocking` statuses.
- For `merge_blocking` output, pass your own `left_key=` / `right_key=` columns, because its `id_lt` numbers restart in every block.

`merge` and `merge_knn` with `k=1` give one candidate per left row. `k` > 1 also shows near-ties. `merge` has no threshold argument: filter on `score` yourself, or pass `drop_sim_threshold=` to `merge_knn`.

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

1. **Dry-run it.** Pass `dry_run=True` with exactly the arguments of the real call. No LLM is called and no checkpoint is written. You get a dict with `llm_calls`, `estimated_input_tokens` (rough, ~4 characters per token), `max_output_tokens_per_call`, `rows_to_send`, `rows_skipped` and `notes`. The count accounts for resumed checkpoints, rows already judged, score-based skips and batching.

   ```python
   plan = lt.merge_k_judge(left, right, on="CompanyName", knn_sbert_model=MODEL, k=3, judge_llm_model="gpt-4o-mini",
                           judge_skip_lower_thresh=0.6, judge_skip_upper_thresh=0.97,
                           embedding_cache="lt_cache", dry_run=True)
   print(plan["llm_calls"], plan["estimated_input_tokens"], plan["max_output_tokens_per_call"], plan["rows_skipped"])
   ```

   A judge dry run needs no API key; a missing key is reported in `notes`. It embeds (runs retrieval) only when `drop_sim_threshold` or a skip threshold makes the count depend on scores, so pass `embedding_cache=` and the real run reuses those embeddings. For `classify_rows` / `transform_rows` dry runs, see [references/llm-workflows.md](references/llm-workflows.md#estimate-cost-before-running).
2. **Price it.** Multiply by the model's current per-token prices; look them up, don't assume them. Output tokens are at most `llm_calls * max_output_tokens_per_call` (50 for the OpenAI judge). Classification replies are short; transforms reply about as long as their input. Dry runs don't cover API *embeddings*: those cost about one token per 4 characters embedded, and cached texts are free.
3. **Pilot on a small sample** (e.g. `left.sample(30, random_state=0)`) and check the output with `lt.match_report` / `llm_raw_response` before scaling up.
4. **Confirm with the user** before the full run when it needs more than about 1,000 LLM calls, or the estimate exceeds about US$5 or their stated budget. Report the call count, the model and the estimated cost.
5. Narrow the judged set first: use a lower `k`, `drop_sim_threshold`, and skip thresholds so pairs that are clearly matches or clearly not skip the LLM. Then dry-run again.

## Long jobs

- For anything that runs more than a couple of minutes, write a script file and run it in the background (`nohup python link_job.py > link_job.log 2>&1 &`) or as a Slurm job. Don't use `python -c` or a notebook cell. Poll the log; don't wait on the process.
- Save intermediate outputs: the candidate table before judging, the judged table after. Then a crash costs only the unfinished stage.
- API functions take `checkpoint_path=` (a JSONL file), `resume=True`, `failure_log_path=` and `retry_policy={"max_retries": 5, "base_sleep_seconds": 15, "jitter_seconds": 2}`. Rerunning the same script with `resume=True` skips finished work. `judge_merge_k_out` also skips rows whose `llm_is_match`, `llm_confidence` and `llm_raw_response` are already filled. Checkpoints are keyed by row position, so resume against the same saved candidate file, never a recomputed one.
- `judge_workers=` (judge) and `api_workers=` (classify/transform) run API calls in parallel. Start at 4 and raise it if there are no rate-limit errors.
- Embedding and search run on CUDA automatically when a GPU is visible.

A complete script template is in [references/llm-workflows.md](references/llm-workflows.md#long-job-template). It saves candidates, gates the paid stage on a `dry_run` call count, resumes from a checkpoint, and ends with a `match_report`. A Slurm variant is there too.

## Gotchas

- **`on` defaults to every column name the two frames share**, IDs included (toy_comp_1/2 share `CompanyID`, `CompanyName` and `Country`). Always pass `on=`, or `left_on=` / `right_on=`.
- **Multi-column keys are serialized into one string per row**, joined by the model tokenizer's separator token (`"TechCorp[SEP]USA"` for MiniLM). API embedding models use `"<SEP>"`; `transform_rows` and API `classify_rows` join with a space. NaN becomes `""`. Order matters, so list left and right columns in corresponding order, and pass `left_on` and `right_on` as the same type (both strings or both lists).
- **`merge_type` is deprecated.** It only warns when keys are not unique, so don't pass it. `merge` never leaves a left row unmatched and may reuse a right row many times.
- **An existing `id_lt` column in either input raises an error**; rename it first.
- **Clustering needs `sim_threshold=`, a cosine similarity** (as in `merge_range`). There is no default; tune it as in [references/recipes.md](references/recipes.md#deduplicate-and-cluster). Old code may pass `cluster_params={"threshold": ...}` instead: that is a cosine *distance* (1 − similarity), so `0.1` means similarity ≥ 0.9. Don't pass both.
- **SLINK (the default) chains:** a row joins a cluster when it is close to *any* member, so a long chain of close pairs can join dissimilar names. Use a high threshold, or `cluster_type="agglomerative"` (average linkage) for tighter clusters on up to tens of thousands of rows; its memory grows with the square of the row count, so use SLINK or `blocking_vars=` beyond that.
- **The LLM judge's output columns are `llm_is_match` (0/1), `llm_confidence`, `llm_raw_response` and `llm_input_payload`**. Rows skipped by the skip thresholds get the sentinel `99` in `llm_is_match` and `llm_confidence`, whether they were skipped for a high score or a low one. `confidence_threshold=` keeps judged rows with `llm_confidence` at or above it, matches *and* non-matches, plus rows skipped for a high score; it drops rows skipped for a low score. Either way, select matches yourself: `llm_is_match == 1`, or `99` with a high `score`.
- **A judge reply that isn't JSON falls back to keywords** (yes / no / match). Anything else counts as a non-match with confidence 0. Replies in a ```` ```json ```` fence, or with text around the JSON, are parsed correctly. Check the pilot for odd replies; see [references/llm-workflows.md](references/llm-workflows.md#check-the-judge-output).
- **`classify_rows` and `transform_rows` use an API only when `openai_key=` or `gemini_key=` is passed explicitly**, e.g. `openai_key=os.environ["OPENAI_API_KEY"]`. Without a key, `classify_rows` treats `model` as a local classifier. Output columns are `clf_preds_<col>` and `transformed_<col>`; multiple columns are joined with `-`.
- **Progress bars fill your context.** Set `LINKTRANSFORMER_PROGRESS=0` before running anything (`false`, `no` and `off` also work). It turns off every bar: embedding, judging, model loading and downloads, and training. It is read on every call, so `os.environ["LINKTRANSFORMER_PROGRESS"] = "0"` inside a script works too. Every function also takes `progress_bar=False`.
- **A `classify_rows` dry run with no key argument reports 0 calls.** Without `openai_key=` / `gemini_key=` it takes the local-classifier path. A `transform_rows` dry run without a key raises. Pass `openai_key=os.environ.get("OPENAI_API_KEY", "")` and `llm_provider="openai"` so the dry run takes the API path even when no key is set.
- **`merge_k_judge` warns** about which retrieval model it resolved. Pass `knn_sbert_model=` (local) or `knn_api_model=` (API embeddings) explicitly; otherwise it retrieves with `model`, which defaults to MiniLM.
- **The embedding cache is keyed by model weights and the exact text.** Changing `on` or the column order changes the text, so those rows are re-embedded. One cache folder can serve every function and both sides of a merge.

## More

- [references/recipes.md](references/recipes.md): multi-column and cross-lingual links, blocking, `merge_range`, aggregation, dedup and clustering, local classification, training.
- [references/llm-workflows.md](references/llm-workflows.md): LLM judging end to end, cost estimation, output checks, `classify_rows` / `transform_rows` with APIs, the long-job script.
