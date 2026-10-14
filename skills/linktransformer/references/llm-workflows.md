# LLM workflows: judging, classification, transforms, long jobs

Snippets marked `# Needs OPENAI_API_KEY` call a paid API. Run a `dry_run=True` estimate and a pilot before any of them on full data, and never put a key in code: read it from the environment. The snippets share this setup:

```python
import json
import os
import pandas as pd
import linktransformer as lt

os.environ.setdefault("LINKTRANSFORMER_PROGRESS", "0")   # no progress bars

left = pd.read_csv(os.path.join(lt.DATA_DIR_PATH, "toy_comp_1.csv"))
right = pd.read_csv(os.path.join(lt.DATA_DIR_PATH, "toy_comp_2.csv"))
MODEL = "dell-research-harvard/lt-wikidata-comp-en"
```

## Retrieve, then judge

The recommended flow has three stages, and only the last one costs money:

1. **Retrieve candidates** with a local model: `merge_knn(k=...)`. This is free, so iterate on `k`, the model and `drop_sim_threshold` here.
2. **Estimate the cost** with `dry_run=True` and pick skip thresholds (next section).
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

Every paid function takes `dry_run=True`. It returns a dict instead of results, calls no LLM and writes no checkpoint. The rows it counts are the ones the real call would send: after rows already judged, rows found in the checkpoint when `resume=True`, score-based skips and batching. So dry-run with exactly the arguments you will use:

```python
cands = lt.merge_knn(left, right, on="CompanyName", model=MODEL, k=3, drop_sim_threshold=0.5,
                     embedding_cache="lt_cache")
lo, hi = 0.6, 0.97                       # judge_skip_lower_thresh, judge_skip_upper_thresh
judge_args = dict(on="CompanyName", judge_llm_model="gpt-4o-mini",
                  judge_skip_lower_thresh=lo, judge_skip_upper_thresh=hi)

plan = lt.judge_merge_k_out(cands, **judge_args, dry_run=True)
print(json.dumps({key: value for key, value in plan.items() if key != "notes"}))
print(plan["notes"])
```

The dict holds:

- `llm_calls` (first attempts; retries come on top) and `estimated_input_tokens` (about 4 characters per token, more for non-ASCII text).
- `max_output_tokens_per_call`: 50 for the OpenAI judge, `None` when no limit is set.
- `rows_total`, `rows_to_send`, `rows_per_call` and `workers`.
- `rows_skipped`, by reason: `already_judged`, `resumed_from_checkpoint`, `outside_judge_score_range`, `null` (transform), and so on.
- `notes`, which include a missing API key. The real call would raise on it; the dry run doesn't.

`merge_k_judge(..., dry_run=True)` adds a `retrieval` entry. It embeds both frames only when `drop_sim_threshold` or a skip threshold makes the count depend on scores. Otherwise the count is exact without embedding anything.

`classify_rows` and `transform_rows` take the API path only when a key *argument* is passed. Without one, a `classify_rows` dry run reports the local classifier (0 calls) and `transform_rows` raises. To dry-run them whether or not a key is set:

```python
news = pd.read_csv(os.path.join(lt.DATA_DIR_PATH, "protest_test.csv"))   # headline, text, label
api_key = os.environ.get("OPENAI_API_KEY", "")   # "" still selects the API path

plan = lt.classify_rows(news, on="text", model="gpt-4o-mini", openai_key=api_key, llm_provider="openai",
                        openai_topic="a protest, strike or riot", batch_size=10, dry_run=True)
print("classify:", plan["llm_calls"], "calls,", plan["estimated_input_tokens"], "input tokens")

plan = lt.transform_rows(right, on="CompanyName", model="gpt-4o-mini", openai_key=api_key, llm_provider="openai",
                         openai_prompt="Expand abbreviations in these company names.", dry_run=True)
print("transform:", plan["llm_calls"], "calls,", plan["estimated_input_tokens"], "input tokens")
```

To get dollars, multiply the tokens by the model's current per-token prices (look them up). Classification replies are a few tokens per row; transforms reply about as long as their input.

Dry runs don't cover API *embeddings*. OpenAI embeddings take about one request per 5,000 characters of text; Gemini takes one request per 100 texts, or one per text when `checkpoint_path` is set. Both bill about one token per 4 characters, and texts already in `embedding_cache` are free.

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

Then summarize the judged pairs. `match_report` reads `llm_is_match`, so run it on the re-parsed frame:

```python
report = lt.match_report(judged, pilot_left, right, on="CompanyName")
print(json.dumps(report["llm"]))         # rows judged / skipped, match rate, confidence, disagreements with the cutoff
print(json.dumps(report["threshold"]))   # score cutoff that agrees best (F1) with the LLM's labels
for s in report["samples"]:
    print(s["score"], s.get("llm_is_match"), s["left"], "<->", s["right"])
```

A high `llm_match_below_threshold` or `llm_non_match_at_or_above_threshold` under `disagreements_at_threshold` means the scores and the judge disagree. Read those pairs before trusting either one.

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
import json
import os

os.environ.setdefault("LINKTRANSFORMER_SILENCE_CITATION", "1")
os.environ.setdefault("LINKTRANSFORMER_PROGRESS", "0")   # no progress bars in the log

import pandas as pd
import linktransformer as lt

OUT = "lt_out"
MAX_LLM_CALLS = int(os.environ.get("MAX_LLM_CALLS", "1000"))   # the budget the user agreed to
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
print(json.dumps(lt.match_report(cands, left, right, on="CompanyName")["score"]), flush=True)

# Stage 2, paid: LLM judge, resumable. The dry run counts only the calls still to make.
judge_args = dict(
    on="CompanyName", judge_llm_model="gpt-4o-mini",
    judge_skip_lower_thresh=0.6, judge_skip_upper_thresh=0.97,
    checkpoint_path=os.path.join(OUT, "judge_checkpoint.jsonl"), resume=True,
    failure_log_path=os.path.join(OUT, "judge_failures.jsonl"),
    retry_policy={"max_retries": 5, "base_sleep_seconds": 15, "jitter_seconds": 2},
    judge_workers=4,
)
plan = lt.judge_merge_k_out(cands, **judge_args, dry_run=True)
print(f"plan: {plan['llm_calls']} calls, ~{plan['estimated_input_tokens']} input tokens", flush=True)
if plan["llm_calls"] > MAX_LLM_CALLS:
    raise SystemExit(f"{plan['llm_calls']} LLM calls exceed MAX_LLM_CALLS={MAX_LLM_CALLS}: confirm with the user first")

judged = lt.judge_merge_k_out(cands, **judge_args)
judged.to_csv(os.path.join(OUT, "judged.csv"), index=False)
print(json.dumps(lt.match_report(judged, left, right, on="CompanyName")["llm"]), flush=True)
print("done", flush=True)
```

To check on it, run `tail -n 5 link_job.log` and `wc -l lt_out/judge_checkpoint.jsonl` (pairs judged so far). Don't block on the process. After a crash, rerun the same command: the dry run then counts only the pairs still to judge. The judged output still needs the [re-parse check](#check-the-judge-output) before you use it.

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
