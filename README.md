# LinkTransformer

[![arXiv](https://img.shields.io/badge/arXiv-2309.00789-b31b1b.svg)](https://arxiv.org/abs/2309.00789)
![LinkTransformer demo](https://github.com/dell-research-harvard/linktransformer/assets/60428420/15162828-b0fb-4ee3-8a0f-fdf3371d10a0)

LinkTransformer is a Python package for semantic record linkage, candidate retrieval, row transformation, clustering, and text classification over tabular data.

- Paper: https://arxiv.org/abs/2309.00789
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

## Provider Notes

- OpenAI key: set `OPENAI_API_KEY` or pass `openai_key`.
- Gemini key: set `GEMINI_API_KEY` or pass `gemini_key`.
- API embedding models and local SBERT models are both supported.
- For multi-column API retrieval, LinkTransformer serializes columns safely using `<SEP>`.

## Test Naming Convention

Tests use `test_lt_*` naming to mirror the package API surface and make workflows discoverable.

## Contributing

Issues and pull requests are welcome.

## License

This project is licensed under the MIT License. See `LICENSE`.

## Maintainers

- Sam Jones (`samuelcaronnajones`)
- Abhishek Arora (`econabhishek`)
- Yiyang Chen (`oooyiyangc`)
