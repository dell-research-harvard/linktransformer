# Train a model from judge labels

Part of the LinkTransformer skill: [SKILL.md](../SKILL.md) covers picking the function and model, thresholds and cost; [llm-workflows.md](llm-workflows.md) covers asking the user, judging pairs and API cost; [recipes.md](recipes.md#train-a-linkage-model) covers `train_model` itself.

The judge (you, or an API model) labels a few hundred candidate pairs, and those labels fine-tune a linkage model. The new model retrieves candidates again, the judge labels the pairs it is least sure about, and the model is trained again. A fixed set of held-out rows, judged in full, measures every round's model and gives the final similarity threshold. What you end with:

- **a model and a threshold** that link the rest of the data, and similar data later, with no judge;
- **a labels file** that retrains the model or extends it to new data.

It pays off when there are too many candidate pairs to judge them all (for you, more than a few hundred; for an API, when the cost or the wait is too much; [When the judge range is too big](llm-workflows.md#when-the-judge-range-is-too-big) lists the other options), when the same kind of linking will come back (new batches, other years, other sources), or when the base model separates matches poorly. For a one-off job of a few hundred pairs, judge the judge range instead ([llm-workflows.md](llm-workflows.md#retrieve-then-judge)).

Training needs `pip install "linktransformer[train]"`. A small model such as `all-MiniLM-L6-v2` trains on a few hundred labels in a minute or two on a laptop CPU; larger base models want a GPU (see [Long jobs](../SKILL.md#long-jobs)).

## Ask the user first

Ask the questions in [llm-workflows.md](llm-workflows.md#ask-the-user-first) (who judges, the API provider and key, what counts as a match), and these, each with your recommendation:

1. **The label budget**: held-out rows, pairs per round and the number of rounds, or a dollar cap for an API judge. If you judge: about 100 held-out rows, 50 to 100 pairs a round and 3 to 4 rounds, which is about 700 to 1,000 pairs to read in all, counting the held-out pairs each new model adds. An API judge: 300 held-out rows and 300 to 500 pairs a round.
2. **The base model** to start from ([Choose a model](../SKILL.md#choose-a-model)). Small models train fastest, and a model already trained for similarity in the right language starts ahead.
3. **Precision or recall** for the final threshold: the best F1, or the most recall at a minimum precision such as 0.95.
4. **Where training runs**: this machine, a GPU, or a cluster.

## Files

| File | What it holds |
|---|---|
| `candidates_r0.csv`, `candidates_r1.csv`, ... | Each round's `merge_knn` output: round 0 from the base model, round r from the model trained after round r − 1. |
| `heldout.csv` | The held-out left rows (`id_lt_x`). Their candidates are judged in full every round and never trained on. |
| `labels.csv` | One decision per line: `id_lt_x`, `id_lt_y`, `is_match`, `confidence`, `reason`. These are the columns of `agent_decisions.csv` in [llm-workflows.md](llm-workflows.md#judge-the-pairs-yourself), so decisions made there can be appended. Keep this file: it is what retrains and extends the model. |
| `loop_models/round1`, ... | The models. Each is trained from the base model on every label about a training row so far. |

**Always retrieve from the full left and right frames, in the same row order.** `id_lt_x` and `id_lt_y` are row positions, so the labels refer to the same rows in every round only if the frames never change.

## Helpers

Save as `loop_helpers.py` next to the job:

```python
"""Helpers for the judge -> train loop of the LinkTransformer skill."""
import os

import numpy as np
import pandas as pd
import linktransformer as lt

KEYS = ["id_lt_x", "id_lt_y"]


def load_labels(path="labels.csv"):
    """Every decision so far, one per pair (the last decision wins)."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=KEYS + ["is_match", "confidence", "reason"])
    return pd.read_csv(path).drop_duplicates(KEYS, keep="last")


def append_labels(decisions, path="labels.csv"):
    decisions[KEYS + ["is_match", "confidence", "reason"]].to_csv(
        path, mode="a", header=not os.path.exists(path), index=False)


def unjudged(pairs, labels):
    """The pairs that have no decision yet."""
    merged = pairs.merge(labels[KEYS], on=KEYS, how="left", indicator=True)
    return merged[merged["_merge"] == "left_only"].drop(columns="_merge")


def evaluate(cands, labels, heldout, min_precision=None):
    """Score cutoff, precision, recall and F1 on the held-out rows' candidates, and the share of held-out rows
    whose best candidate is a match. The cutoff maximizes F1, or recall at precision >= min_precision.
    Recall counts every match found on held-out rows by any round's model."""
    test = cands[cands["id_lt_x"].isin(heldout)].merge(labels[KEYS + ["is_match"]], on=KEYS, how="left")
    if test["is_match"].isna().any():
        raise ValueError(f"{int(test['is_match'].isna().sum())} held-out pairs are not judged yet")
    n_matches = int(((labels["is_match"] == 1) & labels["id_lt_x"].isin(heldout)).sum())
    test = test.sort_values("score", ascending=False)
    tp = test["is_match"].astype(int).cumsum().to_numpy()
    precision = tp / np.arange(1, len(test) + 1)
    recall = tp / max(n_matches, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    if min_precision is None:
        i = int(np.argmax(f1))
    else:
        ok = np.flatnonzero(precision >= min_precision)
        if len(ok) == 0:
            raise ValueError(f"no cutoff reaches precision {min_precision}")
        i = int(ok[-1])
    top1 = test.drop_duplicates("id_lt_x")["is_match"].astype(int).mean()   # sorted, so each row's best
    return {"threshold": round(float(test["score"].iloc[i]), 4), "f1": round(float(f1[i]), 3),
            "precision": round(float(precision[i]), 3), "recall": round(float(recall[i]), 3),
            "top1_match": round(float(top1), 3), "heldout_pairs": len(test)}


def next_batch(cands, labels, heldout, threshold, n):
    """The n unjudged pairs of training rows whose scores are nearest the cutoff."""
    pool = unjudged(cands[~cands["id_lt_x"].isin(heldout)], labels)
    return pool.iloc[np.argsort(np.abs(pool["score"].to_numpy() - threshold), kind="stable")[:n]]


def train_round(cand_files, labels, heldout, base, left_cols, right_cols, name, epochs=5):
    """Train the base model on every confident decision about training rows. Returns the model folder."""
    pairs = pd.concat([pd.read_csv(f) for f in cand_files]).drop_duplicates(KEYS)
    confident = labels[labels["confidence"] > 0.5]
    train = pairs[~pairs["id_lt_x"].isin(heldout)].merge(confident[KEYS + ["is_match"]], on=KEYS)
    return lt.train_model(
        train_data=train, val_data=train, test_data=train,   # train on every label; held-out rows evaluate
        model_path=base, left_col_names=left_cols, right_col_names=right_cols,
        left_id_name="id_lt_x", right_id_name="id_lt_y", label_col_name="is_match",
        training_args={"loss_type": "onlinecontrastive", "num_epochs": epochs, "train_batch_size": 32,
                       "warm_up_perc": 0.1, "model_save_dir": "loop_models", "model_save_name": name},
    )
```

- `onlinecontrastive` learns from the judged non-matches as well as the matches. The judge's non-matches are near misses, the most useful negatives.
- Passing the labels as `train_data`, `val_data` and `test_data` trains on all of them instead of holding 20% back: the held-out rows do the evaluating. `train_model` saves the weights at the end of training, not the best epoch, so the loop compares the rounds itself.
- Decisions with confidence 0.5 or lower (pairs you could not decide, API replies that could not be parsed) are left out of training. Evaluation counts them as non-matches.

## Round 0: hold out rows and label a seed sample

The snippets link the 1947 and 1948 Mexican tariff schedules shipped with the package: product descriptions whose wording changed between the years.

```python
import json
import os
import numpy as np
import pandas as pd
import linktransformer as lt
from loop_helpers import load_labels, unjudged, evaluate, next_batch, train_round

os.environ.setdefault("LINKTRANSFORMER_PROGRESS", "0")
d = pd.read_excel(os.path.join(lt.DATA_DIR_PATH, "es_mexican_products.xlsx")).dropna(subset=["description47", "description48"])
left = d.loc[d["same_text"] == 0, ["description47"]].drop_duplicates().reset_index(drop=True)
right = d[["description48"]].drop_duplicates().reset_index(drop=True)

BASE = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVE = dict(left_on="description47", right_on="description48", k=3, embedding_cache="lt_cache")
LEFT_COLS, RIGHT_COLS = ["description47"], ["description48"]   # key columns as named in the candidate table
N_HELDOUT, PER_ROUND = 150, 100

if not os.path.exists("heldout.csv"):
    rows = np.random.default_rng(0).choice(len(left), size=N_HELDOUT, replace=False)
    pd.DataFrame({"id_lt_x": rows}).to_csv("heldout.csv", index=False)
heldout = set(pd.read_csv("heldout.csv")["id_lt_x"])

if not os.path.exists("candidates_r0.csv"):
    lt.merge_knn(left, right, model=BASE, **RETRIEVE).to_csv("candidates_r0.csv", index=False)
cands = pd.read_csv("candidates_r0.csv")

# 1. Every candidate of the held-out rows
unjudged(cands[cands["id_lt_x"].isin(heldout)], load_labels()).to_csv("judge_heldout_r0.csv", index=False)
```

With `on="CompanyName"` the candidate table names the key columns `CompanyName_x` / `CompanyName_y`, so `LEFT_COLS = ["CompanyName_x"]`; with several `on` columns, list each `_x` column in the same order as `on`. Judge `judge_heldout_r0.csv` ([Judge a batch](#judge-a-batch)), then measure the base model and draw the seed sample: up to 10 pairs from each 0.05-wide score band of the training rows, as in [Find the judge ranges](llm-workflows.md#find-the-judge-ranges):

```python
labels = load_labels()
print(json.dumps(evaluate(cands, labels, heldout)))   # the base model, round 0

pool = unjudged(cands[~cands["id_lt_x"].isin(heldout)], labels)
band = (np.floor(pool["score"] * 20 + 1e-9) / 20).round(2)
seed = pool.assign(band=band).sample(frac=1, random_state=0).groupby("band").head(10).drop(columns="band")
seed.to_csv("judge_r0.csv", index=False)
```

Judge `judge_r0.csv`, then train the first model:

```python
train_round(["candidates_r0.csv"], load_labels(), heldout, BASE, LEFT_COLS, RIGHT_COLS, "round1")
```

## Each later round

Round `r` retrieves with `loop_models/round{r}`, has the judge label the held-out pairs this model found that no earlier one did, and compares every round so far on the same labels:

```python
r = 1
files = [f"candidates_r{i}.csv" for i in range(r + 1)]
if not os.path.exists(files[-1]):
    lt.merge_knn(left, right, model=f"loop_models/round{r}", **RETRIEVE).to_csv(files[-1], index=False)
cands = pd.read_csv(files[-1])
unjudged(cands[cands["id_lt_x"].isin(heldout)], load_labels()).to_csv(f"judge_heldout_r{r}.csv", index=False)
```

Judge `judge_heldout_r{r}.csv`, then:

```python
labels = load_labels()
rounds = pd.DataFrame([evaluate(pd.read_csv(f), labels, heldout) for f in files])
print(rounds.to_string())   # one row per round's model: cutoff, F1, precision, recall, top-1 match rate

cutoff = rounds["threshold"].iloc[-1]
next_batch(cands, labels, heldout, cutoff, PER_ROUND).to_csv(f"judge_r{r}.csv", index=False)
```

Show the user the table. Stop when the budget is spent, or when two rounds in a row gain less than about 0.02 in F1. Otherwise judge `judge_r{r}.csv`, train, and start round `r + 1`:

```python
train_round(files, load_labels(), heldout, BASE, LEFT_COLS, RIGHT_COLS, f"round{r + 1}")
```

- Every round is trained from the base model on all labels so far, so a weak round leaves no trace in the next one.
- `next_batch` takes the pairs whose scores are nearest the current cutoff: the model is least sure about those, and that is where its errors are. On the tariff data, four rounds of 100 such pairs raised F1 on all rows from 0.40 (the base model) to 0.60; four rounds of 100 random pairs reached 0.58.
- Each new model also retrieves some held-out pairs that no earlier model did, and those need judging too. On the tariff data, 450 held-out pairs grew to 725 over four rounds, most of it in the first two.
- Held-out numbers are noisy. On the tariff data, 150 held-out rows held about 130 matches. A single round's F1 was uncertain by about ±0.03, but the gain between two rounds, measured on the same rows, by only about ±0.015. So a gain under 0.02 is within noise. More held-out rows narrow both.

## Judge a batch

Use the judge the user chose for every batch, held-out and training alike, with the same rules for what counts as a match. A label set mixing two judges' standards trains and measures a muddled model.

**You judge.** Read the batch in pages of about 50 pairs and append a decision for every pair you read, following the rules in [Judge the pairs yourself](llm-workflows.md#judge-the-pairs-yourself): read every pair, never fill decisions from a rule or a score, and record pairs you can't decide as `is_match` 0 with confidence 0.5 or lower.

```python
from loop_helpers import append_labels

batch = unjudged(pd.read_csv("judge_r0.csv"), load_labels())
print(len(batch), "pairs left")
print(batch[["id_lt_x", "id_lt_y", "description47", "description48"]].head(50).to_string(index=False))

decisions = [  # (id_lt_x, id_lt_y, is_match, confidence, reason)
]
append_labels(pd.DataFrame(decisions, columns=["id_lt_x", "id_lt_y", "is_match", "confidence", "reason"]))
```

Leave `score` out of what you read: these labels train the model, and decisions nudged by its own scores would teach it its own mistakes.

**An API model judges.** Dry-run each batch and check the cost against the budget before judging it. The checkpoint is keyed by row position, so judge the saved batch file, never a recomputed one:

```python
# Needs OPENAI_API_KEY
from loop_helpers import append_labels

PROMPT = None   # None: the default judge prompt; or the user's rules, as in llm-workflows' "Ask the user first"

def judge_with_api(batch_file, dry_run=False):
    batch = pd.read_csv(batch_file)
    out = lt.judge_merge_k_out(batch, left_on="description47", right_on="description48",
                               judge_llm_model="gpt-4o-mini", llm_prompt=PROMPT,
                               checkpoint_path=batch_file.replace(".csv", ".jsonl"), resume=True,
                               judge_workers=4, dry_run=dry_run)
    if dry_run:
        return out["llm_calls"], out["estimated_input_tokens"]
    append_labels(out.assign(is_match=out["llm_is_match"], confidence=out["llm_confidence"],
                             reason="api: " + out["llm_raw_response"].astype(str)))

print(judge_with_api("judge_r0.csv", dry_run=True))
```

Check the first batch's replies as in [Check the judge output](llm-workflows.md#check-the-judge-output). For an API judge the whole loop can run as one background script, with the round number read from the files that exist and a stop when the dry-run total passes the budget the user agreed to.

## Pick the model and the threshold

At the end, compare every round on the final labels (the table above), and keep the round with the best held-out F1 (or top-1 match rate, if the user will keep only each row's best match), which is not always the last. Its cutoff is the threshold. For a minimum precision:

```python
best = int(rounds["f1"].idxmax())
model = BASE if best == 0 else f"loop_models/round{best}"
cands = pd.read_csv(f"candidates_r{best}.csv")
print(json.dumps(evaluate(cands, labels, heldout)))                      # best F1
print(json.dumps(evaluate(cands, labels, heldout, min_precision=0.95)))  # most recall at precision 0.95
```

If the precision target leaves almost no recall (on the tariff data, 0.95 needs a cutoff of 0.99 and keeps 2% of the matches), tell the user the target is out of reach for this model and offer more rounds or a lower target. If the user will keep only each left row's best match, measure that rule: `evaluate(cands.sort_values("score", ascending=False).drop_duplicates("id_lt_x"), labels, heldout)`. Show the user the held-out pairs on both sides of the cutoff before relying on it:

```python
t = evaluate(cands, labels, heldout)["threshold"]
test = cands[cands["id_lt_x"].isin(heldout)].merge(labels[["id_lt_x", "id_lt_y", "is_match"]])
report = lt.match_report(test.assign(llm_is_match=test["is_match"]), left_on="description47",
                         right_on="description48", threshold=t, n_samples=10)
for s in report["samples"]:
    print(s["score"], s["llm_is_match"], s["left"], "<->", s["right"])
```

Then link everything with the chosen model and threshold, judge-free:

```python
links = lt.merge_knn(left, right, model=model, **RETRIEVE)
links = links[links["score"] >= t]
```

The threshold holds for data like the held-out rows. For data from another source or period, judge 100 or so of its rows in full and run `evaluate` on them before trusting it.

## Use and extend the model

- **Use it** anywhere a model goes: `model="loop_models/round3"` in `merge`, `merge_knn`, `merge_range`, `cluster_rows`, `evaluate_pairs`. It is a sentence-transformers model, so other tools load it too. The threshold was measured for this link; for another use, such as clustering one side, check it again.
- **Extend it** to new data by running the loop on that data in a folder of its own, with the trained model as `BASE`. Its `id_lt_x` / `id_lt_y` refer to the new frames, so its labels can't go into the old `labels.csv`. Each round then trains the trained model further on the new labels. That can make it forget the old data, so also measure the new model on the old held-out rows: retrieve the old frames with it in the old folder, judge the held-out pairs it adds, and `evaluate`. To retrain from the original base on both datasets instead, concatenate the two folders' labeled pairs with their texts (as `train_round` builds them), prefix each folder's ids so they stay distinct, and pass the result to `train_model` as `train_round` does.
- **Share it** only with the user's go-ahead, since publishing is public: `lt.LinkTransformer("loop_models/round3").save_to_hub("user/model-name", private=True)` needs a Hugging Face login, and the model card it writes carries the `linktransformer` tag.
- Pairs that skipped the judge by score (the sentinel `99` in `llm_is_match`) are not labels. Leave them out of `labels.csv`.
