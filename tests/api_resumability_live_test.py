import json
import os

import numpy as np
import pandas as pd
import pytest

import linktransformer as lt
import linktransformer.infer as infer_mod
import linktransformer.utils as utils_mod


OPENAI_KEY = os.getenv("OPENAI_API_KEY")


pytestmark = pytest.mark.skipif(
    OPENAI_KEY is None,
    reason="OPENAI_API_KEY not set for live API resumability tests",
)


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _tagged_checkpoint(base_path: str, tag: str) -> str:
    if base_path.endswith(".jsonl"):
        return base_path[:-6] + f"_{tag}.jsonl"
    return base_path + f"_{tag}.jsonl"


def test_lt_openai_embeddings_checkpoint_resume_live(tmp_path):
    checkpoint_path = tmp_path / "embeds_checkpoint.jsonl"

    strings = ["alpha firm", "beta firm"]
    out_first = utils_mod.infer_embeddings(
        strings=strings,
        model="text-embedding-3-small",
        openai_key=OPENAI_KEY,
        return_numpy=True,
        checkpoint_path=str(checkpoint_path),
        resume=False,
        retry_policy={"max_retries": 2, "base_sleep_seconds": 1},
    )
    rows_first = _read_jsonl(str(checkpoint_path))

    out_second = utils_mod.infer_embeddings(
        strings=strings,
        model="text-embedding-3-small",
        openai_key=OPENAI_KEY,
        return_numpy=True,
        checkpoint_path=str(checkpoint_path),
        resume=True,
        retry_policy={"max_retries": 2, "base_sleep_seconds": 1},
    )
    rows_second = _read_jsonl(str(checkpoint_path))

    assert out_first.shape == out_second.shape
    assert np.allclose(out_first, out_second)
    assert len(rows_first) > 0
    assert len(rows_second) == len(rows_first)


def test_lt_merge_openai_checkpoint_resume_live(tmp_path):
    checkpoint_base = str(tmp_path / "merge_checkpoint.jsonl")

    df1 = pd.DataFrame({"CompanyName": ["Tech Corporation"], "Country": ["USA"]})
    df2 = pd.DataFrame({"CompanyName": ["Tech Corp"], "Country": ["USA"]})

    out_first = lt.merge(
        df1,
        df2,
        on=["CompanyName", "Country"],
        model="text-embedding-3-small",
        openai_key=OPENAI_KEY,
        checkpoint_path=checkpoint_base,
        resume=False,
        retry_policy={"max_retries": 2, "base_sleep_seconds": 1},
    )

    left_cp = _tagged_checkpoint(checkpoint_base, "merge_left_embeddings")
    right_cp = _tagged_checkpoint(checkpoint_base, "merge_right_embeddings")
    left_rows_first = _read_jsonl(left_cp)
    right_rows_first = _read_jsonl(right_cp)

    out_second = lt.merge(
        df1,
        df2,
        on=["CompanyName", "Country"],
        model="text-embedding-3-small",
        openai_key=OPENAI_KEY,
        checkpoint_path=checkpoint_base,
        resume=True,
        retry_policy={"max_retries": 2, "base_sleep_seconds": 1},
    )

    left_rows_second = _read_jsonl(left_cp)
    right_rows_second = _read_jsonl(right_cp)

    assert isinstance(out_first, pd.DataFrame)
    assert out_first[["score"]].equals(out_second[["score"]])
    assert len(left_rows_first) > 0
    assert len(right_rows_first) > 0
    assert len(left_rows_second) == len(left_rows_first)
    assert len(right_rows_second) == len(right_rows_first)


def test_lt_merge_knn_openai_checkpoint_resume_live(tmp_path):
    checkpoint_base = str(tmp_path / "merge_knn_checkpoint.jsonl")

    df1 = pd.DataFrame({"CompanyName": ["Tech Corporation"], "Country": ["USA"]})
    df2 = pd.DataFrame({"CompanyName": ["Tech Corp"], "Country": ["USA"]})

    out_first = lt.merge_knn(
        df1,
        df2,
        on=["CompanyName", "Country"],
        model="text-embedding-3-small",
        openai_key=OPENAI_KEY,
        k=1,
        checkpoint_path=checkpoint_base,
        resume=False,
        retry_policy={"max_retries": 2, "base_sleep_seconds": 1},
    )

    left_cp = _tagged_checkpoint(checkpoint_base, "merge_knn_left_embeddings")
    right_cp = _tagged_checkpoint(checkpoint_base, "merge_knn_right_embeddings")
    left_rows_first = _read_jsonl(left_cp)
    right_rows_first = _read_jsonl(right_cp)

    out_second = lt.merge_knn(
        df1,
        df2,
        on=["CompanyName", "Country"],
        model="text-embedding-3-small",
        openai_key=OPENAI_KEY,
        k=1,
        checkpoint_path=checkpoint_base,
        resume=True,
        retry_policy={"max_retries": 2, "base_sleep_seconds": 1},
    )

    left_rows_second = _read_jsonl(left_cp)
    right_rows_second = _read_jsonl(right_cp)

    assert isinstance(out_first, pd.DataFrame)
    assert out_first[["score"]].equals(out_second[["score"]])
    assert len(left_rows_first) > 0
    assert len(right_rows_first) > 0
    assert len(left_rows_second) == len(left_rows_first)
    assert len(right_rows_second) == len(right_rows_first)


def test_lt_merge_range_openai_checkpoint_resume_live(tmp_path):
    checkpoint_base = str(tmp_path / "merge_range_checkpoint.jsonl")

    df1 = pd.DataFrame({"CompanyName": ["Tech Corporation"], "Country": ["USA"]})
    df2 = pd.DataFrame({"CompanyName": ["Tech Corp"], "Country": ["USA"]})

    out_first = lt.merge_range(
        df1,
        df2,
        on=["CompanyName", "Country"],
        model="text-embedding-3-small",
        openai_key=OPENAI_KEY,
        sim_threshold=0.0,
        checkpoint_path=checkpoint_base,
        resume=False,
        retry_policy={"max_retries": 2, "base_sleep_seconds": 1},
    )

    left_cp = _tagged_checkpoint(checkpoint_base, "merge_range_left_embeddings")
    right_cp = _tagged_checkpoint(checkpoint_base, "merge_range_right_embeddings")
    left_rows_first = _read_jsonl(left_cp)
    right_rows_first = _read_jsonl(right_cp)

    out_second = lt.merge_range(
        df1,
        df2,
        on=["CompanyName", "Country"],
        model="text-embedding-3-small",
        openai_key=OPENAI_KEY,
        sim_threshold=0.0,
        checkpoint_path=checkpoint_base,
        resume=True,
        retry_policy={"max_retries": 2, "base_sleep_seconds": 1},
    )

    left_rows_second = _read_jsonl(left_cp)
    right_rows_second = _read_jsonl(right_cp)

    assert isinstance(out_first, pd.DataFrame)
    assert out_first[["score"]].equals(out_second[["score"]])
    assert len(left_rows_first) > 0
    assert len(right_rows_first) > 0
    assert len(left_rows_second) == len(left_rows_first)
    assert len(right_rows_second) == len(right_rows_first)


def test_lt_classify_rows_checkpoint_resume_live(tmp_path):
    checkpoint_path = tmp_path / "classify_checkpoint.jsonl"

    df = pd.DataFrame(
        {
            "text": [
                "A peaceful protest took place in the city square.",
                "A local bakery introduced a new bread recipe.",
            ]
        }
    )

    out_first = lt.classify_rows(
        df,
        on="text",
        model="gpt-4o-mini",
        num_labels=2,
        openai_key=OPENAI_KEY,
        openai_topic="protests",
        checkpoint_path=str(checkpoint_path),
        resume=False,
        retry_policy={"max_retries": 2, "base_sleep_seconds": 1},
    )
    rows_first = _read_jsonl(str(checkpoint_path))

    out_second = lt.classify_rows(
        df,
        on="text",
        model="gpt-4o-mini",
        num_labels=2,
        openai_key=OPENAI_KEY,
        openai_topic="protests",
        checkpoint_path=str(checkpoint_path),
        resume=True,
        retry_policy={"max_retries": 2, "base_sleep_seconds": 1},
    )
    rows_second = _read_jsonl(str(checkpoint_path))

    assert "clf_preds_text" in out_first.columns
    assert out_first["clf_preds_text"].tolist() == out_second["clf_preds_text"].tolist()
    # predictions are checkpointed in batches (llm_classification_batch); every row must be covered
    covered = sorted(i for rec in rows_first for i in rec.get("indices", [rec.get("index")]))
    assert covered == list(range(len(df)))
    assert len(rows_second) == len(rows_first)  # resuming made no new API calls


def test_lt_transform_rows_checkpoint_resume_live(tmp_path):
    checkpoint_path = tmp_path / "transform_checkpoint.jsonl"

    df = pd.DataFrame(
        {
            "text": [
                "Ths sentnce hs typos.",
                "Anothr txt with errrs.",
            ]
        }
    )

    out_first = lt.transform_rows(
        df,
        on="text",
        openai_key=OPENAI_KEY,
        model="gpt-4o-mini",
        openai_prompt=(
            "Fix spelling mistakes in each input. "
            "Return only a JSON array of corrected strings."
        ),
        batch_size=1,
        checkpoint_path=str(checkpoint_path),
        resume=False,
        retry_policy={"max_retries": 2, "base_sleep_seconds": 1},
        progress_bar=False,
    )
    rows_first = _read_jsonl(str(checkpoint_path))

    out_second = lt.transform_rows(
        df,
        on="text",
        openai_key=OPENAI_KEY,
        model="gpt-4o-mini",
        openai_prompt=(
            "Fix spelling mistakes in each input. "
            "Return only a JSON array of corrected strings."
        ),
        batch_size=1,
        checkpoint_path=str(checkpoint_path),
        resume=True,
        retry_policy={"max_retries": 2, "base_sleep_seconds": 1},
        progress_bar=False,
    )
    rows_second = _read_jsonl(str(checkpoint_path))

    assert "transformed_text" in out_first.columns
    assert out_first["transformed_text"].tolist() == out_second["transformed_text"].tolist()
    assert len(rows_first) == len(df)
    assert len(rows_second) == len(rows_first)


def test_lt_merge_k_judge_checkpoint_resume_live(tmp_path):
    checkpoint_path = tmp_path / "judge_checkpoint.jsonl"

    df1 = pd.DataFrame(
        {
            "CompanyName": ["Tech Corporation"],
            "Country": ["USA"],
        }
    )
    df2 = pd.DataFrame(
        {
            "CompanyName": ["Tech Corp"],
            "Country": ["USA"],
        }
    )

    out_first = infer_mod.merge_k_judge(
        df1=df1,
        df2=df2,
        on=["CompanyName", "Country"],
        model="text-embedding-3-small",
        openai_key=OPENAI_KEY,
        k=1,
        llm_provider="openai",
        judge_llm_model="gpt-4o-mini",
        confidence_threshold=0.0,
        max_retries=1,
        checkpoint_path=str(checkpoint_path),
        resume=False,
        retry_policy={"max_retries": 1, "base_sleep_seconds": 1},
    )
    rows_first = _read_jsonl(str(checkpoint_path))

    out_second = infer_mod.merge_k_judge(
        df1=df1,
        df2=df2,
        on=["CompanyName", "Country"],
        model="text-embedding-3-small",
        openai_key=OPENAI_KEY,
        k=1,
        llm_provider="openai",
        judge_llm_model="gpt-4o-mini",
        confidence_threshold=0.0,
        max_retries=1,
        checkpoint_path=str(checkpoint_path),
        resume=True,
        retry_policy={"max_retries": 1, "base_sleep_seconds": 1},
    )
    rows_second = _read_jsonl(str(checkpoint_path))

    assert len(out_first) == len(out_second) == 1
    assert out_first["llm_is_match"].tolist() == out_second["llm_is_match"].tolist()
    assert out_first["llm_confidence"].tolist() == out_second["llm_confidence"].tolist()
    assert len(rows_first) == 1
    assert len(rows_second) == len(rows_first)


def test_lt_merge_k_judge_retry_policy_zero_compat_live(tmp_path):
    checkpoint_path = tmp_path / "judge_zero_retry_checkpoint.jsonl"

    df1 = pd.DataFrame({"CompanyName": ["Tech Corporation"], "Country": ["USA"]})
    df2 = pd.DataFrame({"CompanyName": ["Tech Corp"], "Country": ["USA"]})

    out = infer_mod.merge_k_judge(
        df1=df1,
        df2=df2,
        on=["CompanyName", "Country"],
        model="text-embedding-3-small",
        openai_key=OPENAI_KEY,
        k=1,
        llm_provider="openai",
        judge_llm_model="gpt-4o-mini",
        max_retries=5,
        checkpoint_path=str(checkpoint_path),
        resume=False,
        retry_policy={"max_retries": 0, "base_sleep_seconds": 0},
    )

    assert len(out) == 1
    assert out.loc[0, "llm_is_match"] == 0
    assert out.loc[0, "llm_confidence"] == 0.0
    assert out.loc[0, "llm_raw_response"] == ""


def test_lt_transform_rows_failure_log_live(tmp_path):
    failure_log_path = tmp_path / "transform_failures.jsonl"

    df = pd.DataFrame({"text": ["simple input"]})

    with pytest.raises(Exception):
        lt.transform_rows(
            df,
            on="text",
            openai_key=OPENAI_KEY,
            model="gpt-not-a-real-model",
            openai_prompt="Return a JSON array.",
            batch_size=1,
            failure_log_path=str(failure_log_path),
            retry_policy={"max_retries": 1, "base_sleep_seconds": 0},
            progress_bar=False,
        )

    rows = _read_jsonl(str(failure_log_path))
    assert len(rows) >= 1
    assert rows[-1]["function"] in {"openai_transform", "transform_column"}
    assert rows[-1]["provider"] == "openai"
