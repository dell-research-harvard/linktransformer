# LinkTransformer recipes

Every snippet runs on the toy data shipped with the package. Swap in your own frames and column names. The snippets share this setup:

```python
import json
import os
import pandas as pd
import linktransformer as lt

os.environ.setdefault("LINKTRANSFORMER_PROGRESS", "0")   # no progress bars
D = lt.DATA_DIR_PATH
firms_a = pd.read_csv(os.path.join(D, "toy_comp_1.csv"))   # CompanyID, CompanyName, Industry, Founded_Year, Country
firms_b = pd.read_csv(os.path.join(D, "toy_comp_2.csv"))   # CompanyID, CompanyName, Revenue..., Num_Employees, Country
```

## Multi-column keys

Pass a list to `on`. The columns are concatenated with the model tokenizer's separator token and embedded as one string. To see exactly what gets embedded:

```python
products_a = pd.read_csv(os.path.join(D, "toy_multi_1.csv"))
products_b = pd.read_csv(os.path.join(D, "toy_multi_2.csv"))
key = ["CompanyName", "ProductDescription"]

print(lt.serialize_columns(products_a.head(2), key, model="all-MiniLM-L6-v2"))
out = lt.merge_knn(products_a, products_b, on=key, model="all-MiniLM-L6-v2", k=2)
print(out[["CompanyName_x", "ProductDescription_x", "CompanyName_y", "ProductDescription_y", "score"]])
```

Include only columns that carry identity: a noisy column (prices, free-text notes) can drag scores down. Compare the score distributions with and without it.

## Different column names, or different languages

Use `left_on` / `right_on`, as two strings or two lists in corresponding order. A multilingual model links across languages:

```python
fr = pd.read_csv(os.path.join(D, "translation_1.csv"))   # "Libellé du Produit"
en = pd.read_csv(os.path.join(D, "translation_2.csv"))   # "Product Label"
out = lt.merge(fr, en, left_on="Libellé du Produit", right_on="Product Label",
               model="dell-research-harvard/lt-un-data-fine-fine-multi")
print(out[["Libellé du Produit", "Product Label", "score"]])
```

For company names across languages, use `dell-research-harvard/lt-wikidata-comp-multi`.

## Fuzzy match inside exact blocks

`merge_blocking` runs `merge` separately inside each group of `blocking_vars`. Use it when a field must agree exactly (country, year, state). It also shrinks the search, since each row is compared only within its block.

```python
out = lt.merge_blocking(firms_a, firms_b, on="CompanyName", blocking_vars=["Country"], model="all-MiniLM-L6-v2")
print(out[["CompanyName_x", "Country_x", "CompanyName_y", "Country_y", "score", "match_status"]])
```

- `match_status` is `matched_block`, `left_only_block` (no right rows in that block) or `right_only_block`. Unmatched rows have NaN `score`.
- `id_lt_x` / `id_lt_y` are row positions *within each block*, so they can't be used to join back. Keep your own ID column for that, and pass it to `match_report`:

  ```python
  report = lt.match_report(out, firms_a, firms_b, on="CompanyName", left_key="CompanyID_x", right_key="CompanyID_y")
  print(json.dumps(report["match_status"]), json.dumps(report["left"]))
  ```
- Rows with a missing value in a blocking column are dropped, because pandas `groupby` drops NaN keys. Fill missing values first (e.g. `fillna("unknown")`) if those rows matter.
- `merge_blocking` takes `embedding_cache=` but not the API checkpoint arguments.

## All matches above a threshold

```python
out = lt.merge_range(firms_a, firms_b, on="CompanyName", model="all-MiniLM-L6-v2", sim_threshold=0.8)
print(out[["CompanyName_x", "CompanyName_y", "score"]])   # unmatched left rows: NaN right side and score
```

## Match new batches against a fixed reference table

Embed the reference once and let the cache serve it on every later call:

```python
for batch in (firms_a.iloc[:3], firms_a.iloc[3:]):
    out = lt.merge_knn(batch, firms_b, on="CompanyName", model="all-MiniLM-L6-v2", k=3,
                       embedding_cache="lt_cache")
```

## Map fine categories to coarse ones

```python
fine = pd.read_csv(os.path.join(D, "fine.csv"))       # Fine Category ID, Fine Category Name, Coarse Category ID
coarse = pd.read_csv(os.path.join(D, "coarse.csv"))   # Coarse Category ID, Coarse Category Name
agg = lt.aggregate_rows(fine, coarse, model="all-MiniLM-L6-v2",
                        left_on="Fine Category Name", right_on="Coarse Category Name")
print(agg[["Fine Category Name", "Coarse Category Name", "score"]])
print("accuracy:", (agg["Coarse Category ID_x"] == agg["Coarse Category ID_y"]).mean())  # toy data has the truth
```

## Deduplicate and cluster

`sim_threshold` is a cosine similarity: rows at least this similar belong together. Every row gets a `cluster` id and a `cluster_size`; a row with no near-duplicate is a cluster of size 1.

```python
firms = pd.read_csv(os.path.join(D, "company_clusters.csv"))   # company_name, cluster_id (ground truth)

clustered = lt.cluster_rows(firms, model="all-MiniLM-L6-v2", on="company_name", sim_threshold=0.85,
                            embedding_cache="lt_cache")
deduped = lt.dedup_rows(firms, model="all-MiniLM-L6-v2", on="company_name", sim_threshold=0.85,
                        keep="medoid", embedding_cache="lt_cache")
print(len(firms), "rows ->", clustered["cluster"].nunique(), "clusters ->", len(deduped), "kept")
print((clustered["cluster_size"] == 1).sum(), "rows without a near-duplicate")
```

Tune the threshold against labeled clusters when you have them, and otherwise by reading a few of the largest clusters:

```python
from sklearn.metrics import adjusted_rand_score

for t in (0.75, 0.8, 0.85, 0.9, 0.95):
    c = lt.cluster_rows(firms, model="all-MiniLM-L6-v2", on="company_name", sim_threshold=t,
                        embedding_cache="lt_cache")
    print(t, c["cluster"].nunique(), c["cluster_size"].max(),
          round(adjusted_rand_score(firms["cluster_id"], c["cluster"]), 3))

big = clustered["cluster"].value_counts().index[0]
print(clustered.loc[clustered["cluster"] == big, "company_name"].head(10).tolist())
```

The default `cluster_type="SLINK"` is single linkage: a row joins a cluster when it is close to any member, so chains of close pairs can join dissimilar names, and the largest clusters show it first. It scales to millions of rows. `cluster_type="agglomerative"` uses average linkage, which keeps clusters tighter, but its memory grows with the square of the row count (about 80 GB for 100,000 rows). `cluster_type="HDBScan"` has no threshold: `cluster_params={"min_cluster_size": 5, "min_samples": 1}`.

Restrict clustering to rows that agree on exact columns with `blocking_vars`; this also splits a large job into small ones:

```python
by_country = lt.cluster_rows(firms_a, model="all-MiniLM-L6-v2", on="CompanyName", sim_threshold=0.85,
                             blocking_vars="Country", embedding_cache="lt_cache")
```

To cluster embeddings you computed yourself (or reuse them across thresholds), use `lt.cluster_embeddings`:

```python
emb = lt.infer_embeddings(firms["company_name"].tolist(), "all-MiniLM-L6-v2", embedding_cache="lt_cache")
for t in (0.8, 0.9):
    print(t, len(set(lt.cluster_embeddings(emb, sim_threshold=t))), "clusters")
```

## Classify rows with a local model

`model` is a Hugging Face sequence-classification checkpoint or a folder saved by `lt.train_clf_model`. A local folder's `label_map.json` turns predictions back into text labels.

```python
news = pd.read_csv(os.path.join(D, "protest_test.csv"))   # headline, byline, text, label (Neither/Protest/Riot)
pred = lt.classify_rows(news.head(16), on="text", model="dell-research-harvard/topic-protests", progress_bar=False)
print(pred[["label", "clf_preds_text"]].value_counts())
```

`num_labels` (default 2) must match the classifier's number of classes.

Other topic classifiers on the Hub follow the same pattern: `dell-research-harvard/topic-<name>` for `antitrust`, `civil_rights`, `crime`, `fire`, `govt_regulation`, `labor_movement`, `obits`, `politics`, `protests`, `sport` and `weather`.

## Train a linkage model

This needs the `[train]` extra. Training data takes one of three forms:

- **Clusters:** a text column and a cluster-id column. Pass `clus_text_col_names=[...]` and `clus_id_col_name=[...]`.
- **Matched pairs:** each row is one left record and its matching right record. Pass `left_col_names`, `right_col_names`, `left_id_name` and `right_id_name`.
- **Labeled pairs:** matched pairs plus a 0/1 column. Pass `label_col_name=`, and use `"loss_type": "onlinecontrastive"` to use the negatives.

```python
path = lt.train_model(
    data=os.path.join(D, "company_clusters.csv"),
    model_path="sentence-transformers/all-MiniLM-L6-v2",
    clus_id_col_name=["cluster_id"],
    clus_text_col_names=["company_name"],
    training_args={"num_epochs": 1, "model_save_dir": "lt_models", "model_save_name": "firms"},
)
out = lt.merge_knn(firms_a, firms_b, on="CompanyName", model=path, k=1)   # use the trained model like any other
```

- `training_args` overrides keys of the default config, `linktransformer/configs/linkage.json` (`num_epochs`, `train_batch_size`, `learning_rate`, `loss_type`, `val_perc`, `model_save_dir`, `model_save_name`, ...). When `training_args` is omitted, `num_epochs` is 10. Without `model_save_dir` the model is saved under `./models/linkage`.
- The returned path is the best checkpoint. Evaluation results and `LT_training_config.json` are saved next to it.
- Real runs take minutes to hours, so run them as background jobs (see [llm-workflows.md](llm-workflows.md#long-job-template)) and on a GPU when you can.

Classifiers train the same way with `lt.train_clf_model`, which returns `(best_model_path, best_metric, label_map)`. Given only `data`, it splits it 70/15/15 into train, eval and test. Labels can be text or integers, and `on` can list several columns, joined with the tokenizer's separator as `classify_rows` does. Two things to get right:

- **Evaluate before the run ends.** The best model is chosen at evaluation steps (every 30 by default). On small data, set `eval_steps` / `save_steps` below the number of training steps, or no checkpoint is saved.
- **Don't keep a second column named `label`.** The label column is renamed to `label` internally.

```python
train = pd.read_csv(os.path.join(D, "protest_train.csv")).sample(120, random_state=0)   # label: Neither / Protest / Riot

best_path, best_f1, label_map = lt.train_clf_model(
    data=train, model="distilroberta-base", on=["headline", "text"], label_col_name="label",
    epochs=1, batch_size=8, eval_steps=5, save_steps=5,
    data_dir="lt_clf_data", model_save_dir="lt_clf",
)
pred = lt.classify_rows(train.head(8), on=["headline", "text"], model=best_path, num_labels=len(label_map),
                        progress_bar=False)
print(label_map)
print(pred[["label", "clf_preds_headline-text"]])
```

Predictions come back as the original text labels, read from the `label_map.json` saved with the model.
