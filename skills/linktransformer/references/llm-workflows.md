# LLM workflows: judging, classification, transforms, long jobs

Part of the LinkTransformer skill: [SKILL.md](../SKILL.md) covers picking the function and model, thresholds and cost; [recipes.md](recipes.md) covers linking patterns without a judge; [judge-train-loop.md](judge-train-loop.md) trains a model from judge labels when there are too many pairs to judge.

Snippets marked `# Needs OPENAI_API_KEY` call a paid API. Run a `dry_run=True` estimate and a pilot before any of them on full data, and never put a key in code: read it from the environment. The snippets share this setup:

```python
import json
import os
import numpy as np
import pandas as pd
import linktransformer as lt

os.environ.setdefault("LINKTRANSFORMER_PROGRESS", "0")   # no progress bars

left = pd.read_csv(os.path.join(lt.DATA_DIR_PATH, "toy_comp_1.csv"))
right = pd.read_csv(os.path.join(lt.DATA_DIR_PATH, "toy_comp_2.csv"))
MODEL = "dell-research-harvard/lt-wikidata-comp-en"
```

## Retrieve, then judge

The recommended flow has five steps. Only judging costs anything, and only the pairs in the judge range are judged:

1. **Retrieve candidates** with a local model: `merge_knn(k=...)`, saved to a file. This is free, so do it before asking anything about judging, and iterate on `k`, the model and `drop_sim_threshold` here. While the model is still being chosen, `all-MiniLM-L6-v2` is enough to size the job.
2. **Ask the user** who judges (you or an API model), the provider and key for an API model, what counts as a match, and whether very similar pairs may be accepted without judging. Include the candidate counts by score band, so the user sees how much judging may lie ahead ([Ask the user first](#ask-the-user-first)).
3. **Find the judge ranges** from a small calibration sample: an *auto-accept* range of very high scores (only if the user agreed), a *never-judge* range of low scores, and the *judge* range in between. Confirm them with the user, with the number of pairs to judge and, for an API judge, the cost ([Find the judge ranges](#find-the-judge-ranges)). If the judge range holds more pairs than the chosen judge can take, offer the options in [When the judge range is too big](#when-the-judge-range-is-too-big).
4. **Judge the judge range**, yourself ([Judge the pairs yourself](#judge-the-pairs-yourself)) or with `judge_merge_k_out` and an API model (pilot first, then everything with a checkpoint).
5. **Check the output, audit an API judge and pick the matches** ([Check the judge output](#check-the-judge-output), [Audit the judge](#audit-the-judge)).

`merge_k_judge(df1, df2, ...)` runs retrieval and judging in one call. That suits small jobs, but it recomputes the candidates on every call and doesn't checkpoint API embeddings. For big jobs, save the candidates and call `judge_merge_k_out` yourself.

Judge parameters worth knowing:

- `judge_llm_model` (required). `llm_provider="auto"` routes names containing `gemini` to Gemini and everything else to OpenAI. Keys come from `OPENAI_API_KEY` / `GEMINI_API_KEY`.
- `judge_on`, or `judge_left_on` / `judge_right_on`: columns sent to the judge. They can differ from the retrieval columns, e.g. retrieve on the name, then judge on name + industry + city. For each column the judge reads `<col>_x` / `<col>_y` when those exist, and otherwise `<col>`.
- `judge_skip_lower_thresh` / `judge_skip_upper_thresh`: the never-judge and auto-accept cutoffs. Pairs with `score <= lower` or `score >= upper` skip the LLM and get the sentinel `99`; `None` turns a cutoff off.
- `llm_prompt` replaces the system prompt. Keep asking for JSON with `is_match` and `confidence`. It is sent with every call, so keep it short: in one trial the prompt was 387 of the ~460 input tokens per call. The judge sees each pair as `{"left": {<column>: <value>, ...}, "right": {...}}`, keyed by column name, so judge columns whose names say what they hold. `llm_params` takes `temperature` (default 0), `max_tokens` (default 50 for OpenAI) and `request_timeout` (default 15 s).
- `judge_workers` sets parallel calls; `checkpoint_path`, `resume`, `failure_log_path` and `retry_policy` are covered below.
- In `merge_k_judge`, pass `knn_sbert_model=` (local retrieval) or `knn_api_model=` (API embeddings) explicitly; `model=` is only a fallback.

## Ask the user first

After retrieving candidates (free) and before judging, ask these in one message (with your question tool if you have one), each with your recommendation. If the user hasn't named a model, put the model question from [Choose a model](../SKILL.md#choose-a-model) in the same message. Give the candidate counts by score band from the retrieval: with dense data, such as product titles that differ only by version or edition, most candidates can land in the judge range.

1. **Who judges the pairs?** You, the coding agent: no API key and no API cost, but every pair uses your context, so it suits up to a few hundred pairs in the judge range. Or an API model: costs money, runs unattended with checkpoints, and suits thousands of pairs. The calibration below gives the size of the judge range; if it turns out too big for the chosen judge, come back with the options in [When the judge range is too big](#when-the-judge-range-is-too-big).
2. **For an API model: which provider and model?** OpenAI (e.g. `gpt-4o-mini`) or Gemini (a current Gemini model; needs `pip install "linktransformer[gemini]"`). Then check whether the key is set, without printing it: `bool(os.getenv("OPENAI_API_KEY"))` or `bool(os.getenv("GEMINI_API_KEY"))`. If it is not, ask the user to set it in the environment the job runs in (their shell profile, a `.env` file they load, or the job script's environment) and to tell you when it is done. Don't ask them to paste a key into the chat, and never write one into code, files or logs.
3. **What counts as a match?** The default is the same real-world entity. Ask about the cases that matter for their data, such as a parent and its subsidiary, branches, a company before and after a rename, or a product and its variants. Apply the answer when you judge, and give it to an API judge in `llm_prompt`, keeping the JSON instruction: `"Decide whether LEFT and RIGHT refer to the same real-world entity. <the user's rules>. Return ONLY compact JSON with keys is_match (0 or 1) and confidence (0 to 1)."`
4. **May very similar pairs be accepted without judging?** An auto-accept range saves judging the pairs that are almost always matches, but some users want every pair checked. Pairs in the never-judge range are always treated as non-matches.

## Find the judge ranges

Take up to 10 random candidate pairs from each 0.05-wide score band (about 100 to 200 pairs in all), have the chosen judge label them, and read off the ranges. Fewer than 10 per band often misses the few matches in the lower bands, and the never-judge range then silently drops them.

A pair's rank among its left row's candidates matters as much as its score. In the Amazon–Google products trial, a left row's top candidate (above 0.55) was a match 85% of the time, and its 2nd to 5th candidates only 10 to 16% of the time. So with `k` > 1, split each band's 10 pairs between top candidates and the rest:

```python
cands = pd.read_csv("candidates.csv")      # saved merge_knn output
cands["band"] = (np.floor(cands["score"].astype(float) * 20 + 1e-9) / 20).round(2)   # 0.05-wide score bands
cands["top"] = cands.groupby("id_lt_x")["score"].rank(ascending=False, method="first") == 1   # each left row's best
print(cands.groupby(["band", "top"]).size().unstack(fill_value=0).sort_index(ascending=False).to_string())
per_cell = 10 if cands["top"].all() else 5          # k=1: every pair is a top candidate
calib = (cands.sample(frac=1, random_state=0).groupby(["band", "top"]).head(per_cell)
         .sort_values("score", ascending=False))
cols = ["id_lt_x", "id_lt_y", "band", "top", "score", "CompanyName_x", "CompanyName_y", "Country_x", "Country_y"]
print(calib[cols].to_string(index=False))
```

Label the calibration pairs with the judge the user chose:

- **You judge:** read them and record your decisions as in [Judge the pairs yourself](#judge-the-pairs-yourself). They count toward the final judging too. Then `labels = pd.read_csv("agent_decisions.csv")[["id_lt_x", "id_lt_y", "is_match"]]`.
- **API judge:** dry-run, then judge only the sample: `pilot = lt.judge_merge_k_out(calib, on="CompanyName", judge_llm_model="gpt-4o-mini", checkpoint_path="calib_judge.jsonl", resume=True)` and `labels = pilot[["id_lt_x", "id_lt_y", "llm_is_match"]].rename(columns={"llm_is_match": "is_match"})`.

Then propose the cutoffs. Bands at the bottom in which every sampled pair was a non-match become the never-judge range, and bands at the top in which every sampled pair was a match become the auto-accept range, each less one band as a safety margin:

```python
by_band = calib.merge(labels, on=["id_lt_x", "id_lt_y"]).groupby("band")["is_match"].agg(["mean", "size"])
print(by_band.sort_index(ascending=False).to_string())          # match rate of the sampled pairs in each band
all_no = (by_band["mean"] == 0).cumprod().astype(bool).to_numpy()              # from the lowest band up
all_yes = (by_band["mean"] == 1)[::-1].cumprod()[::-1].astype(bool).to_numpy()  # from the highest band down
no_bands, yes_bands = by_band.index[all_no], by_band.index[all_yes]
seen_match, seen_non_match = bool((by_band["mean"] > 0).any()), bool((by_band["mean"] < 1).any())
lo = float(no_bands[-1]) if seen_match and len(no_bands) >= 2 else None        # never judge: score <= lo
hi = float(yes_bands[1]) if seen_non_match and len(yes_bands) >= 2 else None   # auto-accept: score >= hi
# hi = None                                                 # if the user wants every pair judged

accept = cands["score"] >= hi if hi is not None else pd.Series(False, index=cands.index)
never = cands["score"] <= lo if lo is not None else pd.Series(False, index=cands.index)
print(f"auto-accept (score >= {hi}): {accept.sum()} | never judge (score <= {lo}): {never.sum()} | "
      f"judge: {(~accept & ~never).sum()} of {len(cands)}")
```

The never-judge range is where matches can be lost without anyone seeing them, so check the pairs just below `lo` before you propose it. Label 20 more from the two bands under the cutoff. If any is a match, lower `lo` by a band and check again. In the trial, this found a match at 0.54 under a proposed cutoff of 0.55:

```python
if lo is not None:
    near = cands[(cands["score"] > lo - 0.10) & (cands["score"] <= lo)].drop(calib.index, errors="ignore")
    extra = near.sample(min(20, len(near)), random_state=1)
    print(extra[cols].to_string(index=False))   # label these like the calibration pairs
```

Then estimate how many matches the judge range holds, for top candidates and the rest. The user needs these numbers to choose among the options when the range is too big:

```python
def in_judge_range(df):
    keep = pd.Series(True, index=df.index)
    if lo is not None:
        keep &= df["score"] > lo
    if hi is not None:
        keep &= df["score"] < hi
    return keep

rate = calib.merge(labels, on=["id_lt_x", "id_lt_y"]).groupby(["band", "top"])["is_match"].mean()
pairs = cands[in_judge_range(cands)].groupby(["band", "top"]).size()
estimate = pd.DataFrame({"pairs_to_judge": pairs.groupby(level="top").sum(),
                         "estimated_matches": (pairs * rate.reindex(pairs.index)).groupby(level="top").sum().round()})
print(estimate)   # rows: top=True (each left row's best candidate) and top=False (the rest)
```

The estimate is rough, a few labels per cell, but it shows where the matches are. If the sample holds no matches at all (or no non-matches), it cannot place a cutoff, so none is proposed: tell the user, and check the model and what counts as a match with them before judging everything. Show the user the three ranges with their pair counts, a few pairs on each side of each cutoff (`lt.match_report(cands, left, right, on="CompanyName", threshold=hi, n_samples=10)["samples"]`, and the same with `lo`), and the cost: the number of pairs you would read, or the API dry run below. Judge nothing more until the user confirms or adjusts the cutoffs.

## When the judge range is too big

Dense data can put thousands of pairs in the judge range. In the Amazon–Google products trial, 5,600 of 6,800 candidates fell between the cutoffs, because titles that differ only by version or edition score like true matches. A lower `k` or a higher `lo` couldn't fix that without losing most matches, and no agent can read thousands of pairs. Tell the user the number of pairs, then these options with the matches each keeps (from the estimate above):

1. **An API model judges the range and you audit it.** Pilot the API judge on your calibration labels ([Pilot](#pilot)), run it on the whole range with a checkpoint, then read a few hundred of its calls, picked where judges go wrong ([Audit the judge](#audit-the-judge)), and override the ones you disagree with. In the trial, `gpt-4.1-mini` made about 5,300 calls for US$1.30, and the agent agreed with 49 of 50 randomly audited matches.
2. **Train a model from judge labels** ([judge-train-loop.md](judge-train-loop.md)). You or an API model label a few hundred pairs per round, and the trained model links the rest with no judge. Best when the same kind of linking will come back.
3. **Judge only each left row's top candidate** and treat the rest of the range as non-matches. This reads at most one pair per left row. It loses the matches that weren't the top candidate (the `top=False` row of the estimate), including every second match of a left row with several. Judge them from their own saved file, since checkpoints are keyed by row position: `cands[cands["top"] & in_judge_range(cands)].to_csv("top_candidates.csv", index=False)`.

To keep pairs away from the judge, drop them from the candidate file. Don't fill in `99`: every run decides `99` rows again from its own skip thresholds.

## Estimate cost before running

Every paid function takes `dry_run=True`. It returns a dict instead of results, calls no LLM and writes no checkpoint. The rows it counts are the ones the real call would send: after rows already judged, rows found in the checkpoint when `resume=True`, score-based skips and batching. So dry-run with exactly the arguments you will use:

```python
cands = lt.merge_knn(left, right, on="CompanyName", model=MODEL, k=3, drop_sim_threshold=0.5,
                     embedding_cache="lt_cache")
lo, hi = 0.6, 0.97                       # the confirmed never-judge and auto-accept cutoffs (None turns one off)
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

Test an API judge on pairs whose answers you know before paying for the full run. If you labeled the calibration sample yourself (or the pool from [Compare candidate models](../SKILL.md#compare-candidate-models)), pilot on those labels, and measure the judge where it will work, on the pairs in the judge range:

```python
# Needs OPENAI_API_KEY
PROMPT = ("Decide whether LEFT and RIGHT refer to the same real-world entity. <the user's rules>. "
          "Return ONLY compact JSON with keys is_match (0 or 1) and confidence (0 to 1).")
known = calib.merge(labels, on=["id_lt_x", "id_lt_y"])
known = known[in_judge_range(known)]                # only pairs the judge would see
pilot = lt.judge_merge_k_out(known.drop(columns="is_match"), on="CompanyName", judge_llm_model="gpt-4o-mini",
                             llm_prompt=PROMPT, checkpoint_path="pilot_judge.jsonl", resume=True, progress_bar=False)
pred, truth = pilot["llm_is_match"] == 1, known["is_match"] == 1
tp = (pred & truth).sum()
print(f"{len(known)} pairs | agreement {(pred == truth).mean():.2f} | "
      f"precision {tp / max(pred.sum(), 1):.2f} | recall {tp / max(truth.sum(), 1):.2f}")
print(lt.match_report(pilot)["llm"]["tokens"])      # tokens billed (releases after 0.1.19)
```

- **Report precision and recall, not only agreement.** Easy pairs inflate agreement. In the trial, a first prompt agreed with the agent's labels 91.5% of the time, but its precision was 0.71, because it matched different titles from the same series. A stricter prompt reached 0.905.
- **If you rewrite the prompt from these results**, a second pilot on the same labels is optimistic. Tune on half the labels and report on the other half.
- **Price the full run from the pilot's tokens** (`llm_input_tokens`, `llm_output_tokens`), not only from the dry run's estimate. Thinking models bill their reasoning as output: a current Gemini Flash judge used 90 to 270 output tokens per pair for a one-line JSON reply.

Without labels, pilot on a sample of left rows and read the results:

```python
# Needs OPENAI_API_KEY
pilot_left = left.sample(min(30, len(left)), random_state=0)
pilot = lt.merge_k_judge(pilot_left, right, on="CompanyName", knn_sbert_model=MODEL, k=3,
                         drop_sim_threshold=0.5, judge_llm_model="gpt-4o-mini",
                         judge_skip_lower_thresh=lo, judge_skip_upper_thresh=hi, progress_bar=False)
print(pilot[["CompanyName_x", "CompanyName_y", "score", "llm_is_match", "llm_confidence", "llm_raw_response"]])
```

Either way, read the pairs the judge rejected with a high `score`, and those it accepted with a low one. Adjust the prompt, the judged columns or the thresholds before scaling up.

## Judge the pairs yourself

When the user chooses you as the judge, you decide the pairs yourself: no key and no API cost, but every pair uses your context. That suits up to a few hundred pairs in the judge range. For more, see [When the judge range is too big](#when-the-judge-range-is-too-big).

Read only the judge range, and record decisions in the judge's columns so `match_report` and the match filter below work unchanged:

```python
# 1. Candidates (free), saved once: decisions refer to their id_lt_x / id_lt_y
lt.merge_knn(left, right, on="CompanyName", model=MODEL, k=3, embedding_cache="lt_cache").to_csv("candidates.csv", index=False)

# 2. The judge range, minus pairs already decided (so a rerun continues where you stopped)
cands = pd.read_csv("candidates.csv")
lo, hi = 0.6, 0.97                       # the confirmed cutoffs; None turns one off
in_range = pd.Series(True, index=cands.index)
if lo is not None:
    in_range &= cands["score"] > lo
if hi is not None:
    in_range &= cands["score"] < hi
gray = cands[in_range]
done = pd.read_csv("agent_decisions.csv") if os.path.exists("agent_decisions.csv") else pd.DataFrame(columns=["id_lt_x", "id_lt_y"])
todo = gray.merge(done[["id_lt_x", "id_lt_y"]], on=["id_lt_x", "id_lt_y"], how="left", indicator=True)
todo = todo[todo["_merge"] == "left_only"].drop(columns="_merge")
print(len(gray), "pairs in the judge range,", len(todo), "still to judge")

# 3. Read a batch: only the columns needed to decide
cols = ["id_lt_x", "id_lt_y", "score", "CompanyName_x", "CompanyName_y", "Industry", "Country_x", "Country_y"]
print(todo[cols].head(50).to_string(index=False))
```

After reading a batch, append your decisions, one per pair you read:

```python
decisions = [  # (id_lt_x, id_lt_y, is_match, confidence, reason)
    (0, 0, 1, 0.95, "TechCorp / Tech Corporation: same name with a legal suffix, same country"),
]
pd.DataFrame(decisions, columns=["id_lt_x", "id_lt_y", "is_match", "confidence", "reason"]).to_csv(
    "agent_decisions.csv", mode="a", header=not os.path.exists("agent_decisions.csv"), index=False)
```

Repeat steps 2 and 3 until nothing is left, then put the decisions into the judge's columns. Skipped pairs get the sentinel `99`, as with the API judge:

```python
dec = pd.read_csv("agent_decisions.csv").drop_duplicates(["id_lt_x", "id_lt_y"], keep="last")
judged = cands.merge(dec, on=["id_lt_x", "id_lt_y"], how="left")
skipped = ~in_range
judged["llm_is_match"] = judged["is_match"].where(~skipped, 99)
judged["llm_confidence"] = judged["confidence"].where(~skipped, 99.0)
judged["llm_raw_response"] = ("agent: " + judged["reason"]).where(~skipped, "99")
judged = judged.drop(columns=["is_match", "confidence", "reason"])
print(json.dumps(lt.match_report(judged, left, right, on="CompanyName")["llm"]))
```

How to judge:

- Ask the same question as the API judge: do the LEFT and RIGHT records refer to the same real-world entity? Use the fields shown and general knowledge (abbreviations, legal suffixes, translations, former names).
- Read every pair you record. Never fill decisions from a rule or a script, such as a string-similarity cutoff: that is another threshold, not a judgment.
- When the fields don't settle it, record `is_match` 0 with confidence 0.5 or lower and say what is missing in `reason`. List those pairs for the user.
- Work in batches of about 50 pairs and append after each batch, so an interruption loses at most one batch.
- Tell the user how many pairs you judged and how many were uncertain. `llm_raw_response` starts with `agent:` on the pairs you decided.

To hand the rest to an API model, pass `judged` to `judge_merge_k_out` with the same skip thresholds: it judges only the pairs that have no decision yet.

## Check the judge output

The judge reads `is_match` and `confidence` from JSON replies, including replies in a ```` ```json ```` fence or with text around them. A reply that isn't JSON falls back to keywords (yes / no / match), and anything else counts as a non-match with confidence 0. So look for replies that never stated `is_match`:

```python
llm_ran = pilot["llm_is_match"] != 99   # 99: skipped by score, no LLM call
odd = pilot[llm_ran & ~pilot["llm_raw_response"].astype(str).str.contains("is_match")]
print(len(odd), "replies without an is_match field")
print(odd["llm_raw_response"].head().tolist())
```

If there are many, raise `llm_params={"max_tokens": ...}` (a cut-off reply loses its fields) or make the prompt stricter. Then summarize the judged pairs:

```python
report = lt.match_report(pilot, pilot_left, right, on="CompanyName")
print(json.dumps(report["llm"]))         # rows judged / skipped, match rate, confidence, disagreements with the cutoff
print(json.dumps(report["threshold"]))   # score cutoff that agrees best (F1) with the LLM's labels
for s in report["samples"]:
    print(s["score"], s.get("llm_is_match"), s["left"], "<->", s["right"])
```

A high `llm_match_below_threshold` or `llm_non_match_at_or_above_threshold` under `disagreements_at_threshold` means the scores and the judge disagree. Read those pairs before trusting either one.

Then pick the matches: pairs judged a match, plus the auto-accept range when the user agreed to one:

```python
auto = (pilot["llm_is_match"] == 99) & (pilot["score"] >= hi) if hi is not None else False
matches = pilot[(pilot["llm_is_match"] == 1) | auto]
best = matches.sort_values("score", ascending=False).drop_duplicates("id_lt_x")   # at most one match per left row
```

`confidence_threshold=` keeps judged rows with `llm_confidence >= threshold`, matches and non-matches alike, plus rows skipped for a high score. It drops rows skipped for a low score. It doesn't select matches, so you still need the filter above.

## Audit the judge

After a full API run, read a few hundred of its calls yourself, picked where judges go wrong, and override the ones you disagree with. Its own `llm_confidence` is a poor guide: `gpt-4.1-mini` reported 0.9 for 92% of its matches. The cutoffs below are starting points; adjust them to the data:

```python
judged = pd.read_csv("lt_out/judged.csv")
judged["rank"] = judged.groupby("id_lt_x")["score"].rank(ascending=False, method="first")
match = judged["llm_is_match"] == 1
matches_per_left = judged["id_lt_x"].map(judged[match].groupby("id_lt_x").size()).fillna(0)
audit = pd.concat([
    judged[match].sample(min(50, int(match.sum())), random_state=0),                  # random matches: precision
    judged[match & ((judged["rank"] >= 3) | (judged["score"] < (lo or 0) + 0.10))],   # far down the list, or barely judged
    judged[(judged["llm_is_match"] == 0) & (judged["rank"] == 1) & (judged["score"] >= 0.85)],  # rejected top candidates
    judged[match & (matches_per_left >= 3)],                                           # left rows with many matches
]).drop_duplicates(["id_lt_x", "id_lt_y"])
print(len(audit), "pairs to audit")
```

Read them in batches and record decisions in `agent_decisions.csv`, as in [Judge the pairs yourself](#judge-the-pairs-yourself). If the user's reading budget is smaller, drop the later groups first, but keep the random sample. Then let your decisions override the judge's:

```python
dec = pd.read_csv("agent_decisions.csv").drop_duplicates(["id_lt_x", "id_lt_y"], keep="last")
final = judged.merge(dec, on=["id_lt_x", "id_lt_y"], how="left")
mine = final["is_match"].notna()
final.loc[mine, "llm_is_match"] = final.loc[mine, "is_match"].astype(int)
final.loc[mine, "llm_confidence"] = final.loc[mine, "confidence"]
final.loc[mine, "llm_raw_response"] = "agent: " + final.loc[mine, "reason"].astype(str)
final = final.drop(columns=["is_match", "confidence", "reason"])
```

Tell the user your agreement on the random sample (an estimate of the judge's precision), what kinds of pairs it got wrong, and how many calls you overrode. In the trial, the judge's errors were edition, bundle, pack and OEM differences, and similar titles from one publisher.

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
| `merge_k_judge`, `judge_merge_k_out` | one line per LLM call, with its token counts (after 0.1.19) | row position in the candidate table |
| `classify_rows` (API) | batches of labels | row position |
| `transform_rows` | batches of outputs | position among non-null rows |
| `merge`, `merge_knn`, `merge_range` with API embeddings | embedding chunks, in `<path>_<function>_<side>_embeddings.jsonl` | chunk position |

- Positions must refer to the same input, so resume against the same saved file in the same row order.
- After 0.1.19, pairs skipped by `judge_skip_*_thresh` are not checkpointed: every run decides them again from its own thresholds, so a rerun with a wider judge range judges the newly included pairs. 0.1.19 checkpoints them as `99`, and later versions ignore those lines.
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
LO, HI = 0.6, 0.97   # the never-judge and auto-accept cutoffs the user confirmed; None turns one off
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
    judge_skip_lower_thresh=LO, judge_skip_upper_thresh=HI,
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
print(json.dumps(lt.match_report(judged, left, right, on="CompanyName")["llm"]), flush=True)   # with "tokens" after 0.1.19
print("done", flush=True)
```

To check on it, run `tail -n 5 link_job.log` and `wc -l lt_out/judge_checkpoint.jsonl` (LLM calls so far; 0.1.19 also counts score-skipped pairs). Don't block on the process. After a crash, rerun the same command: the dry run then counts only the pairs still to judge.

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
