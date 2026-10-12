"""Model card (README.md) templates for LinkTransformer models and classifiers pushed to the Hugging Face Hub."""
import os
import re
from typing import Iterable, List, Optional, Union

CITATION = """@inproceedings{arora-dell-2024-linktransformer,
    title = "{L}ink{T}ransformer: A Unified Package for Record Linkage with Transformer Language Models",
    author = "Arora, Abhishek and Dell, Melissa",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-demos.21/",
    doi = "10.18653/v1/2024.acl-demos.21",
    pages = "221--231"
}"""

_LOSS_NAMES = {"supcon": "supervised contrastive (SupCon)", "onlinecontrastive": "online contrastive"}


class ModelCardTemplate:
    __TAGS__ = ["linktransformer", "sentence-transformers", "sentence-similarity", "tabular-classification"]
    __DEFAULT_VARS__ = {
        "{MODEL_DESCRIPTION}": "",
        "{EVALUATION}": "<!--- Describe how your model was evaluated -->",
    }

    __MODEL_CARD__ = """---
{FRONT_MATTER}
---

# {MODEL_NAME}

This is a [LinkTransformer](https://linktransformer.github.io/) model. It is a [sentence-transformers](https://www.SBERT.net) model, so it works with both packages.
It maps texts such as names, addresses and short descriptions to a {NUM_DIMENSIONS}-dimensional dense vector space, and is designed for record linkage (entity matching), deduplication, clustering and aggregation with the LinkTransformer package.

{BASE_MODEL_AND_LANGUAGE}

{MODEL_DESCRIPTION}

## Usage (LinkTransformer)

```
pip install -U linktransformer
```

```python
import linktransformer as lt
import pandas as pd

# Two dataframes whose key column is written differently, e.g. company names
df1 = pd.read_csv("df1.csv")  # left dataframe
df2 = pd.read_csv("df2.csv")  # right dataframe

# For each row of df1, find the closest row of df2; the "score" column holds the cosine similarity
df_merged = lt.merge(df1, df2, on="CompanyName", model="{MODEL_NAME}")
```

LinkTransformer also supports k-nearest-neighbour and threshold-based merges (`lt.merge_knn`, `lt.merge_range`), deduplication (`lt.dedup_rows`), clustering (`lt.cluster_rows`) and aggregation (`lt.aggregate_rows`). See the [documentation](https://linktransformer.github.io/).

## Usage (Sentence-Transformers)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("{MODEL_NAME}")
embeddings = model.encode(["Tech Corporation", "Tech Corp"])
```

{TRAINING_SECTION}

## Evaluation Results

{EVALUATION}

## Full Model Architecture

```
{FULL_MODEL_STR}
```

## Citing & Authors

```bibtex
{CITATION}
```
"""

    __TRAINING_SECTION__ = """## Training

{TRAINING_DETAILS}

You can train your own LinkTransformer model with `lt.train_model`. Any sentence-transformers model can be the backbone, and so can any other Hugging Face model if you set `add_pooling_layer=True` in `training_args`. For example, to link Mexican tariff descriptions from 1947 and 1948 (an example from the paper):

```python
import linktransformer as lt

saved_model_path = lt.train_model(
    model_path="hiiamsid/sentence_similarity_spanish_es",
    data="mexican_tariffs.csv",
    left_col_names=["description47"],
    right_col_names=["description48"],
    left_id_name=["tariffcode47"],
    right_id_name=["tariffcode48"],
    training_args={"num_epochs": 1},
)
```"""


class ClassificationModelCardTemplate:
    __TAGS__ = ["linktransformer", "transformers", "text-classification", "tabular-classification"]
    __DEFAULT_VARS__ = {
        "{MODEL_DESCRIPTION}": "",
        "{EVALUATION}": "<!--- Describe how your model was evaluated -->",
    }

    __MODEL_CARD__ = """---
{FRONT_MATTER}
---

# {MODEL_NAME}

This is a text classification model from the [LinkTransformer](https://linktransformer.github.io/) ecosystem. It is a standard Hugging Face Transformers sequence classification model that assigns texts to the labels below.

{BASE_MODEL_AND_LANGUAGE}

{LABEL_MAP_SECTION}

{MODEL_DESCRIPTION}

## Usage (LinkTransformer)

```
pip install -U linktransformer
```

```python
import linktransformer as lt

df_clf_output = lt.classify_rows(df, on=["col_of_interest"], model="{MODEL_NAME}")
```

## Usage (Transformers)

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="{MODEL_NAME}")
classifier("Protesters gathered outside city hall on Monday.")
```

{TRAINING_SECTION}

## Evaluation Results

{EVALUATION}

Evaluation is typically based on metrics like accuracy, F1-score, precision, and recall.

## Citing & Authors

```bibtex
{CITATION}
```
"""

    __TRAINING_SECTION__ = """## Training

You can train your own LinkTransformer classifier with `lt.train_clf_model`:

```python
import linktransformer as lt

best_model_path, best_metric, label_map = lt.train_clf_model(
    data="path_to_dataset.csv",
    model="distilroberta-base",
    on=["text_column_name"],
    label_col_name="label_column_name",
    lr=5e-5,
    batch_size=16,
    epochs=3,
)
```

More information on configurations can be found in the [LinkTransformer documentation](https://linktransformer.github.io/)."""


def _as_list(value: Union[str, Iterable[str], None]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [v for v in value if v]


def _is_hub_id(name: Optional[str]) -> bool:
    """True for Hugging Face Hub ids like "org/model" or "distilroberta-base", False for local paths."""
    return bool(name) and not os.path.exists(name) and re.fullmatch(r"[\w.-]+(/[\w.-]+)?", name) is not None


def _front_matter(pipeline_tag: str, library_name: str, tags: List[str], languages: List[str],
                  base_model: Optional[str], datasets: List[str]) -> str:
    lines = [f"library_name: {library_name}", f"pipeline_tag: {pipeline_tag}"]
    if _is_hub_id(base_model):
        lines.append(f"base_model: {base_model}")
    if languages:
        lines += ["language:"] + [f"- {lang}" for lang in languages]
    lines += ["tags:"] + [f"- {tag}" for tag in tags]
    if datasets:
        lines += ["datasets:"] + [f"- {d}" for d in datasets]
    return "\n".join(lines)


def _base_model_and_language(base_model: Optional[str], languages: List[str], verb: str) -> str:
    parts = []
    if base_model:
        shown = f"[{base_model}](https://huggingface.co/{base_model})" if _is_hub_id(base_model) else f"`{base_model}`"
        parts.append(f"It was {verb} from {shown}.")
    if languages:
        parts.append(f"Language(s): {', '.join(languages)}.")
    return " ".join(parts)


def _fill(template: str, values: dict) -> str:
    for key, value in values.items():
        template = template.replace(key, "" if value is None else str(value))
    leftover = re.findall(r"\{[A-Z][A-Z_]+\}", template)  # placeholders, not BibTeX's {L}ink{T}ransformer
    if leftover:
        raise ValueError(f"Unfilled model card placeholders: {sorted(set(leftover))}")
    # collapse the blank lines left by empty optional sections
    return re.sub(r"\n{3,}", "\n\n", template).strip() + "\n"


def _training_details(training_config: Optional[dict]) -> str:
    if not training_config:
        return "If this model was trained with LinkTransformer, its training configuration is saved as `LT_training_config.json` in this repository."
    details = []
    loss = training_config.get("loss_type")
    if loss:
        details.append(f"{_LOSS_NAMES.get(loss, loss)} loss")
    for key, label in [("num_epochs", "epochs"), ("train_batch_size", "batch size"), ("learning_rate", "learning rate")]:
        if training_config.get(key) is not None:
            details.append(f"{label} {training_config[key]}")
    summary = f" ({', '.join(details)})" if details else ""
    return (f"This model was trained with the LinkTransformer package{summary}. "
            "The full training configuration is saved as `LT_training_config.json` in this repository.")


def render_linktransformer_card(model_name: str, num_dimensions: int, architecture: str,
                                base_model: Optional[str] = None, languages=None, description: Optional[str] = None,
                                train_datasets=None, training_config: Optional[dict] = None) -> str:
    """Render the README.md for a LinkTransformer (sentence embedding) model."""
    languages, datasets = _as_list(languages), _as_list(train_datasets)
    return _fill(ModelCardTemplate.__MODEL_CARD__, {
        "{FRONT_MATTER}": _front_matter("sentence-similarity", "sentence-transformers", ModelCardTemplate.__TAGS__,
                                        languages, base_model, datasets),
        "{TRAINING_SECTION}": ModelCardTemplate.__TRAINING_SECTION__,
        "{TRAINING_DETAILS}": _training_details(training_config),
        "{BASE_MODEL_AND_LANGUAGE}": _base_model_and_language(base_model, languages, "fine-tuned"),
        "{MODEL_DESCRIPTION}": description or ModelCardTemplate.__DEFAULT_VARS__["{MODEL_DESCRIPTION}"],
        "{EVALUATION}": ModelCardTemplate.__DEFAULT_VARS__["{EVALUATION}"],
        "{NUM_DIMENSIONS}": num_dimensions,
        "{FULL_MODEL_STR}": architecture,
        "{CITATION}": CITATION,
        "{MODEL_NAME}": model_name,
    })


def render_classifier_card(model_name: str, base_model: Optional[str] = None, languages=None,
                           description: Optional[str] = None, label_map: Optional[dict] = None,
                           train_datasets=None) -> str:
    """Render the README.md for a LinkTransformer text classifier."""
    languages, datasets = _as_list(languages), _as_list(train_datasets)
    label_section = ""
    if label_map:
        label_section = "Labels are mapped to integers as follows:\n\n" + "\n".join(f"- {k}: {v}" for k, v in label_map.items())
    return _fill(ClassificationModelCardTemplate.__MODEL_CARD__, {
        "{FRONT_MATTER}": _front_matter("text-classification", "transformers", ClassificationModelCardTemplate.__TAGS__,
                                        languages, base_model, datasets),
        "{TRAINING_SECTION}": ClassificationModelCardTemplate.__TRAINING_SECTION__,
        "{BASE_MODEL_AND_LANGUAGE}": _base_model_and_language(base_model, languages, "fine-tuned"),
        "{LABEL_MAP_SECTION}": label_section,
        "{MODEL_DESCRIPTION}": description or ClassificationModelCardTemplate.__DEFAULT_VARS__["{MODEL_DESCRIPTION}"],
        "{EVALUATION}": ClassificationModelCardTemplate.__DEFAULT_VARS__["{EVALUATION}"],
        "{CITATION}": CITATION,
        "{MODEL_NAME}": model_name,
    })
