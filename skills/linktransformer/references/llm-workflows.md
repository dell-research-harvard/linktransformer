# LLM workflows: judging, classification, transforms, long jobs

Snippets marked `# Needs OPENAI_API_KEY` call a paid API. Run the cost estimate and a pilot before any of them on full data, and never put a key in code: read it from the environment. The snippets share this setup:

```python
import os
import pandas as pd
import linktransformer as lt

left = pd.read_csv(os.path.join(lt.DATA_DIR_PATH, "toy_comp_1.csv"))
right = pd.read_csv(os.path.join(lt.DATA_DIR_PATH, "toy_comp_2.csv"))
MODEL = "dell-research-harvard/lt-wikidata-comp-en"
```

## Retrieve, then judge

The recommended flow has three stages, and only the last one costs money:

1. **Retrieve candidates** with a local model: `merge_knn(k=...)`. This is free, so iterate on `k`, the model and `drop_sim_threshold` here.
2. **Estimate the cost** and pick skip thresholds (next section).
3. **Judge** the saved candidates with `judge_merge_k_out`. Pilot on a sample first, then run everything with a checkpoint.

`merge_k_judge(df1, df2, ...)` runs stages 1 and 3 in one call. That suits small jobs, but it recomputes the candidates on every call and doesn't checkpoint API embeddings. For big jobs, save the candidates and call `judge_merge_k_out` yourself.

Judge parameters worth knowing:

- `judge_llm_model` (required). `llm_provider="auto"` routes names containing `gemini` to Gemini and everything else to OpenAI. Keys come from `OPENAI_API_KEY` / `GEMINI_API_KEY`.
- `judge_on`, or `judge_left_on` / `judge_right_on`: columns sent to the judge. They can differ from the retrieval columns, e.g. retrieve on the name, then judge on name + industry + city. For each column the judge reads `<col>_x` / `<col>_y` when those exist, and otherwise `<col>`.
- `judge_skip_lower_thresh` / `judge_skip_upper_thresh`: pairs with `score <= lower` or `score >= upper` skip the LLM and get the sentinel `99`.
- `llm_prompt` replaces the system prompt. Keep asking for JSON with `is_match` and `confidence`. `llm_params` takes `temperature` (default 0), `max_tokens` (default 50 for OpenAI) and `request_timeout` (default 15 s).
- `judge_workers` sets parallel calls; `checkpoint_path`, `resume`, `failure_log_path` and `retry_policy` are covered below.
- In `merge_k_judge`, pass `knn_sbert_model=` (local retrieval) or `knn_api_model=` (API embeddings) explicitly; `model=` is only a fallback.

## Estimate cost before running

```python
cands = lt.merge_knn(left, right, on="CompanyName", model=MODEL, k=3, drop_sim_threshold=0.5,
                     embedding_cache="lt_cache")
lo, hi = 0.6, 0.97                       # judge_skip_lower_thresh, judge_skip_upper_thresh
to_judge = cands[(cands["score"] > lo) & (cands["score"] < hi)]
n_calls = len(to_judge)                  # one LLM call per judged pair

# about 4 characters per token: system prompt (~350 chars) + JSON payload, compact JSON reply
payload_chars = to_judge["CompanyName_x"].astype(str).str.len() + to_judge["CompanyName_y"].astype(str).str.len() + 60
input_tokens = int(((payload_chars + 350) / 4).sum())
output_tokens = 20 * n_calls
print(f"{n_calls} calls, ~{input_tokens} input and ~{output_tokens} output tokens")
print(cands["score"].describe())         # check how many pairs the skip thresholds remove
```

Multiply the tokens by the model's current per-token prices (look them up) to get dollars. Calls for the other functions:

| Function | LLM calls |
|---|---|
| `merge_k_judge` | pairs from `merge_knn(k, drop_sim_threshold)` with `lower < score < upper` |
| `judge_merge_k_out` | same, over `candidates_df`, minus rows that already have `llm_*` values |
| `classify_rows` (API) | `ceil(n_rows / batch_size)`, default `batch_size=128` texts per call |
| `transform_rows` | `ceil(n_non_null_rows / batch_size)`, default `batch_size=50` |
| OpenAI embeddings | about one request per 5,000 characters of text; billed per token. Cached texts are free. |
| Gemini embeddings | one request per 100 texts (one per text when `checkpoint_path` is set) |

For classification, input tokens are about the total characters of the texts / 4. For transforms, budget about as many output tokens as input tokens.

Confirm with the user before a full run above roughly 1,000 calls or US$5, or their stated budget.

## Pilot

```python
# Needs OPENAI_API_KEY
pilot_left = left.sample(min(30, len(left)), random_state=0)
pilot = lt.merge_k_judge(pilot_left, right, on="CompanyName", knn_sbert_model=MODEL, k=3,
                         drop_sim_threshold=0.5, judge_llm_model="gpt-4o-mini",
                         judge_skip_lower_thresh=lo, judge_skip_upper_thresh=hi, progress_bar=False)
print(pilot[["CompanyName_x", "CompanyName_y", "score", "llm_is_match", "llm_confidence", "llm_raw_response"]])
```

Read the pairs the judge rejected with a high `score`, and those it accepted with a low one. Adjust the prompt, the judged columns or the thresholds before scaling up.

## Check the judge output

In 0.2.0 the reply parser mis-reads a reply wrapped in a ```` ```json ```` fence as a match, and some models (often Gemini) add those fences. Re-parse the raw replies and compare:

```python
import json
import re

def reparse_is_match(raw):
    """is_match from a raw judge reply, tolerating code fences; None if the reply has no readable JSON."""
    found = re.search(r"\{.*\}", str(raw), flags=re.S)
    try:
        return int(json.loads(found.group(0))["is_match"])
    except (AttributeError, ValueError, KeyError, TypeError):
        return None

judged = pilot.copy()
llm_ran = judged["llm_is_match"] != 99
judged.loc[llm_ran, "llm_is_match"] = judged.loc[llm_ran, "llm_raw_response"].map(reparse_is_match)
print("unreadable replies:", judged.loc[llm_ran, "llm_is_match"].isna().sum())
```

Then decide matches yourself, counting pairs skipped for a high score as matches:

```python
matches = judged[(judged["llm_is_match"] == 1) | ((judged["llm_is_match"] == 99) & (judged["score"] >= hi))]
best = matches.sort_values("score", ascending=False).drop_duplicates("id_lt_x")   # at most one match per left row
```

`confidence_threshold=` filters on `llm_confidence >= threshold`, which also keeps every skipped row (`99`), including those skipped for a low score. Prefer the explicit filter above.

## Classify rows with an LLM

The API path runs only when a key is passed explicitly; otherwise `classify_rows` loads `model` as a local classifier.

```python
# Needs OPENAI_API_KEY
news = pd.read_csv(os.path.join(lt.DATA_DIR_PATH, "protest_test.csv"))   # headline, text, label
labeled = lt.classify_rows(
    news.head(20), on="text", model="gpt-4o-mini", openai_key=os.environ["OPENAI_API_KEY"],
    openai_topic="a protest, strike or riot",   # prompt: "Determine whether each text is about <topic>."
    label_map={"Yes": 1, "No": 0}, num_labels=2,
    batch_size=10, api_workers=2, progress_bar=False,
    checkpoint_path="lt_out/classify.jsonl", resume=True,
)
print(labeled[["label", "clf_preds_text"]].value_counts())
```

- Each call carries `batch_size` texts and must return exactly that many labels, so lower `batch_size` for long texts.
- With `openai_prompt=` (your own system prompt) pass a `label_map` whose keys are the labels the prompt asks for. `num_labels` must equal `len(label_map)`. If any reply contains a label missing from `label_map`, *all* predictions come back as raw text labels, with a warning.
- For Gemini, pass `gemini_key=os.environ["GEMINI_API_KEY"]` and a Gemini chat model (the repo's live tests use `"gemini-3.5-flash-lite"`).

## Clean text with an LLM before linking

```python
# Needs OPENAI_API_KEY
cleaned = lt.transform_rows(
    right, on="CompanyName", model="gpt-4o-mini", openai_key=os.environ["OPENAI_API_KEY"],
    openai_prompt="Expand abbreviations and fix OCR errors in these company names. Keep legal-form words.",
    batch_size=50, progress_bar=False, checkpoint_path="lt_out/transform.jsonl", resume=True,
)
out = lt.merge_knn(left, cleaned, left_on="CompanyName", right_on="transformed_CompanyName", model=MODEL, k=1)
```

`model` defaults to `"gpt-4o"`, so pass a cheaper one explicitly. Null cells are not sent and stay null in the output column. Read a sample of `transformed_*` against the original before linking on it.

## Checkpoints and resume

| Function | `checkpoint_path` records | Resume key |
|---|---|---|
| `merge_k_judge`, `judge_merge_k_out` | one line per judged pair | row position in the candidate table |
| `classify_rows` (API) | batches of labels | row position |
| `transform_rows` | batches of outputs | position among non-null rows |
| `merge`, `merge_knn`, `merge_range` with API embeddings | embedding chunks, in `<path>_<function>_<side>_embeddings.jsonl` | chunk position |

- Positions must refer to the same input, so resume against the same saved file in the same row order.
- For embeddings, prefer `embedding_cache=`: it is keyed by text, works for API models too, and is shared across functions and runs.
- `failure_log_path` collects one JSON line per call that failed after retries. Rate limits, timeouts and 5xx errors are retried with exponential backoff (`base_sleep_seconds * 2**attempt`). Other 4xx errors (a bad key, an unknown model) fail immediately.
- A call that still fails after its retries stops the whole function. Fix the cause and rerun with `resume=True`: finished rows are not paid for twice.

## Long job template

Save as `link_job.py` and start it in the background:

```bash
nohup python link_job.py > link_job.log 2>&1 &
```

```python
# Needs OPENAI_API_KEY
"""Retrieve candidates locally, then judge them with an LLM. Rerun the same command to resume."""
import os

os.environ.setdefault("LINKTRANSFORMER_SILENCE_CITATION", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import pandas as pd
import linktransformer as lt

OUT = "lt_out"
os.makedirs(OUT, exist_ok=True)
left = pd.read_csv(os.path.join(lt.DATA_DIR_PATH, "toy_comp_1.csv"))
right = pd.read_csv(os.path.join(lt.DATA_DIR_PATH, "toy_comp_2.csv"))

# Stage 1, free: candidates from a local model. Saved once, because the judge checkpoint
# refers to row positions in this file.
cand_path = os.path.join(OUT, "candidates.csv")
if not os.path.exists(cand_path):
    cands = lt.merge_knn(left, right, on="CompanyName", model="dell-research-harvard/lt-wikidata-comp-en",
                         k=3, drop_sim_threshold=0.5, embedding_cache=os.path.join(OUT, "emb_cache"))
    cands.to_csv(cand_path, index=False)
cands = pd.read_csv(cand_path)
print(f"{len(cands)} candidate pairs", flush=True)

# Stage 2, paid: LLM judge, resumable.
judged = lt.judge_merge_k_out(
    cands, on="CompanyName", judge_llm_model="gpt-4o-mini",
    judge_skip_lower_thresh=0.6, judge_skip_upper_thresh=0.97,
    checkpoint_path=os.path.join(OUT, "judge_checkpoint.jsonl"), resume=True,
    failure_log_path=os.path.join(OUT, "judge_failures.jsonl"),
    retry_policy={"max_retries": 5, "base_sleep_seconds": 15, "jitter_seconds": 2},
    judge_workers=4, progress_bar=False,
)
judged.to_csv(os.path.join(OUT, "judged.csv"), index=False)
print("done", flush=True)
```

To check on it, run `tail -n 5 link_job.log` and `wc -l lt_out/judge_checkpoint.jsonl` (pairs judged so far). Don't block on the process.

On a Slurm cluster, submit the same script from a shell where the API key is already exported. `sbatch` passes the submitting environment to the job by default, so the key never has to be written to disk. Add partition and account flags as the cluster requires:

```bash
#!/bin/bash
#SBATCH --job-name=lt-link
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=link_job_%j.log
python link_job.py
```

The GPU line is optional: it speeds up embedding and search, which use CUDA when it is available. LLM calls are network-bound, so raise `judge_workers` / `api_workers` instead.
