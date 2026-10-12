import builtins
import json
import types
import concurrent.futures

import numpy as np
import pandas as pd
import pytest

import linktransformer.utils as utils_mod


def test_lt_load_model_success_returns_model_instance(monkeypatch):
    class FakeLinkTransformer:
        def __init__(self, model_path):
            self.model_path = model_path

    monkeypatch.setattr(utils_mod, "LinkTransformer", FakeLinkTransformer)

    model = utils_mod.load_model("dummy/model/path")
    assert isinstance(model, FakeLinkTransformer)
    assert model.model_path == "dummy/model/path"


def test_lt_load_model_failure_raises_value_error(monkeypatch):
    class FakeLinkTransformer:
        def __init__(self, model_path):
            raise RuntimeError("boom")

    monkeypatch.setattr(utils_mod, "LinkTransformer", FakeLinkTransformer)

    with pytest.raises(ValueError, match="Can't load the model"):
        utils_mod.load_model("broken/path")


def test_lt_predict_rows_with_openai_requires_v1(monkeypatch):
    monkeypatch.setattr(utils_mod.openai, "__version__", "0.28.1")

    with pytest.raises(ValueError, match="Requires OpenAI API version 1.0.0 or higher"):
        utils_mod.predict_rows_with_openai(
            strings_col=["sample"],
            model="gpt-3.5-turbo",
            openai_key="dummy",
            openai_params={},
        )


def test_lt_gemini_import_error_message(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "google.generativeai":
            raise ImportError("No module named google.generativeai")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="google-generativeai"):
        utils_mod.infer_embeddings_with_gemini(
            strings=["hello"],
            model="gemini-embedding-001",
            api_key="dummy",
        )


def test_lt_serialize_columns_respects_sep_token_for_openai_model(monkeypatch):
    df = pd.DataFrame(
        {
            "left": ["alpha", "beta"],
            "right": ["one", "two"],
        }
    )

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("AutoTokenizer.from_pretrained should not be called for OpenAI embedding models when sep_token is provided")

    monkeypatch.setattr(utils_mod.transformers.AutoTokenizer, "from_pretrained", _fail_if_called)

    out = utils_mod.serialize_columns(
        df,
        ["left", "right"],
        sep_token="<SEP>",
        model="text-embedding-3-small",
    )

    assert out == ["alpha<SEP>one", "beta<SEP>two"]


def test_lt_serialize_columns_uses_tokenizer_sep_when_not_provided(monkeypatch):
    df = pd.DataFrame({"a": ["x"], "b": ["y"]})

    class FakeTokenizer:
        sep_token = "@@"

    monkeypatch.setattr(
        utils_mod.transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )

    out = utils_mod.serialize_columns(df, ["a", "b"], model="sentence-transformers/all-MiniLM-L6-v2")
    assert out == ["x@@y"]


def test_lt_serialize_columns_respects_explicit_sep_for_non_api_model(monkeypatch):
    df = pd.DataFrame({"a": ["x"], "b": ["y"]})

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("Tokenizer lookup should be skipped when sep_token is explicitly provided")

    monkeypatch.setattr(utils_mod.transformers.AutoTokenizer, "from_pretrained", _fail_if_called)

    out = utils_mod.serialize_columns(
        df,
        ["a", "b"],
        sep_token="|",
        model="sentence-transformers/all-MiniLM-L6-v2",
    )
    assert out == ["x|y"]


def test_lt_serialize_columns_falls_back_to_default_sep_when_tokenizer_missing(monkeypatch):
    df = pd.DataFrame({"a": ["x"], "b": ["y"]})

    def _raise(*args, **kwargs):
        raise RuntimeError("tokenizer unavailable")

    monkeypatch.setattr(utils_mod.transformers.AutoTokenizer, "from_pretrained", _raise)

    out = utils_mod.serialize_columns(df, ["a", "b"], model="sentence-transformers/all-MiniLM-L6-v2")
    assert out == ["x</s>y"]


def test_lt_serialize_columns_handles_null_values(monkeypatch):
    df = pd.DataFrame({"a": ["x", None], "b": [None, "y"]})

    out = utils_mod.serialize_columns(df, ["a", "b"], sep_token="|", model="text-embedding-3-small")
    assert out == ["x|", "|y"]


def test_lt_serialize_columns_uses_linktransformer_tokenizer_sep(monkeypatch):
    class FakeLinkTransformer:
        def __init__(self):
            self.tokenizer = type("Tok", (), {"sep_token": "<LTSEP>"})()

    monkeypatch.setattr(utils_mod, "LinkTransformer", FakeLinkTransformer)

    df = pd.DataFrame({"a": ["x"], "b": ["y"]})
    out = utils_mod.serialize_columns(df, ["a", "b"], model=FakeLinkTransformer())
    assert out == ["x<LTSEP>y"]


def test_lt_openai_embedding_model_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OpenAI embedding models require an API key"):
        utils_mod.infer_embeddings(
            strings=["alpha"],
            model="text-embedding-3-small",
            openai_key=None,
        )


def test_lt_openai_embedding_model_uses_env_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")

    captured = {}

    class _FakeEmbeddings:
        def create(self, input, model):
            captured["model"] = model
            captured["input"] = input
            class _Item:
                def __init__(self):
                    self.embedding = [0.1, 0.2]
            class _Response:
                data = [_Item()]
            return _Response()

    class _FakeOpenAIClient:
        def __init__(self, api_key=None, timeout=None):
            captured["api_key"] = api_key
            self.embeddings = _FakeEmbeddings()

    monkeypatch.setattr(utils_mod.openai, "api_key", None)
    monkeypatch.setattr(utils_mod.openai, "__version__", "1.3.0")
    monkeypatch.setattr(utils_mod.openai, "embeddings", _FakeEmbeddings())

    out = utils_mod.infer_embeddings(
        strings=["alpha"],
        model="text-embedding-3-small",
        openai_key=None,
        return_numpy=True,
    )

    assert out.shape == (1, 2)
    assert captured["model"] == "text-embedding-3-small"
    assert captured["input"] == ["alpha"]
    assert utils_mod.openai.api_key == "env-openai-key"


def test_lt_predict_rows_with_openai_resume_from_checkpoint(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "clf_checkpoint.jsonl"
    checkpoint_path.write_text(
        "\n".join(
            [
                json.dumps({"kind": "openai_classification_row", "index": 0, "response": "Yes"}),
                json.dumps({"kind": "openai_classification_row", "index": 1, "response": "No"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class _FakeClient:
        pass

    monkeypatch.setattr(utils_mod.openai, "__version__", "1.3.0")
    monkeypatch.setattr(utils_mod.openai, "OpenAI", lambda **kwargs: _FakeClient())

    calls = {"n": 0}

    def _fake_completion(client, text, model, openai_key, openai_topic, openai_prompt, openai_params):
        calls["n"] += 1
        return "Yes", 3

    monkeypatch.setattr(utils_mod, "get_completion_from_messages", _fake_completion)

    out = utils_mod.predict_rows_with_openai(
        strings_col=["a", "b", "c"],
        model="gpt-4o-mini",
        openai_key="dummy",
        openai_topic="topic",
        label_dict={"Yes": 1, "No": 0},
        checkpoint_path=str(checkpoint_path),
        resume=True,
        retry_policy={"max_retries": 1, "base_sleep_seconds": 0},
    )

    assert out == [1, 0, 1]
    assert calls["n"] == 1


def test_lt_predict_rows_with_openai_logs_failure(monkeypatch, tmp_path):
    failure_log_path = tmp_path / "clf_failures.jsonl"

    class _FakeClient:
        pass

    monkeypatch.setattr(utils_mod.openai, "__version__", "1.3.0")
    monkeypatch.setattr(utils_mod.openai, "OpenAI", lambda **kwargs: _FakeClient())

    def _always_fail(*args, **kwargs):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(utils_mod, "get_completion_from_messages", _always_fail)

    with pytest.raises(RuntimeError, match="rate limited"):
        utils_mod.predict_rows_with_openai(
            strings_col=["a"],
            model="gpt-4o-mini",
            openai_key="dummy",
            openai_topic="topic",
            failure_log_path=str(failure_log_path),
            retry_policy={"max_retries": 1, "base_sleep_seconds": 0},
        )

    lines = failure_log_path.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[-1])
    assert payload["function"] == "predict_rows_with_openai"
    assert payload["provider"] == "openai"


def test_lt_transform_column_resume_from_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "transform_checkpoint.jsonl"
    checkpoint_path.write_text(
        json.dumps({"kind": "transform_column_row", "index": 0, "value": "old-a"}) + "\n",
        encoding="utf-8",
    )

    df = pd.DataFrame({"text": ["a", "b", "c"]})
    calls = {"n": 0}

    def _transform_fn(batch, params):
        calls["n"] += 1
        return [f"new-{item}" for item in batch]

    out = utils_mod.transform_column(
        df,
        column="text",
        transform_fn=_transform_fn,
        fn_kwargs={},
        chunk_size=1,
        output_column="tx",
        progress_bar=False,
        checkpoint_path=str(checkpoint_path),
        resume=True,
    )

    assert out["tx"].tolist() == ["old-a", "new-b", "new-c"]
    assert calls["n"] == 2


def test_lt_gemini_infer_embeddings_resume_checkpoint(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "gemini_embed_checkpoint.jsonl"
    checkpoint_path.write_text(
        json.dumps({"kind": "gemini_embedding_item", "index": 0, "embedding": [0.1, 0.2]}) + "\n",
        encoding="utf-8",
    )

    calls = {"n": 0}

    def _fake_gemini(strings, model, api_key, return_numpy=True):
        calls["n"] += 1
        assert strings == ["second"]
        return np.array([[0.3, 0.4]], dtype=np.float32)

    monkeypatch.setattr(utils_mod, "infer_embeddings_with_gemini", _fake_gemini)

    out = utils_mod.infer_embeddings(
        strings=["first", "second"],
        model="gemini-embedding-001",
        gemini_key="g-key",
        checkpoint_path=str(checkpoint_path),
        resume=True,
        return_numpy=True,
        retry_policy={"max_retries": 1, "base_sleep_seconds": 0},
    )

    assert out.shape == (2, 2)
    assert np.allclose(out[0], np.array([0.1, 0.2], dtype=np.float32))
    assert np.allclose(out[1], np.array([0.3, 0.4], dtype=np.float32))
    assert calls["n"] == 1


def test_lt_predict_rows_with_llm_openai_batch_and_throttle(monkeypatch):
    monkeypatch.setattr(utils_mod.openai, "__version__", "1.3.0")
    monkeypatch.setattr(utils_mod.openai, "OpenAI", lambda **kwargs: object())

    call_counter = {"n": 0}

    def _fake_resilient(client, request_kwargs):
        call_counter["n"] += 1
        payload = json.loads(request_kwargs["messages"][1]["content"])
        labels = ["Yes" for _ in payload]
        msg = types.SimpleNamespace(content=json.dumps(labels))
        choice = types.SimpleNamespace(message=msg)
        return types.SimpleNamespace(choices=[choice])

    sleep_calls = []

    monkeypatch.setattr(utils_mod, "_openai_chat_completion_resilient", _fake_resilient)
    monkeypatch.setattr(utils_mod.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    out = utils_mod.predict_rows_with_llm(
        strings_col=["a", "b", "c", "d", "e"],
        model="gpt-4o-mini",
        openai_key="dummy",
        llm_provider="openai",
        openai_prompt="Classify",
        label_dict={"Yes": 1, "No": 0},
        batch_size=2,
        throttle_seconds=0.25,
        progress_bar=False,
        retry_policy={"max_retries": 1, "base_sleep_seconds": 0},
    )

    assert out == [1, 1, 1, 1, 1]
    assert call_counter["n"] == 3
    assert sleep_calls == [0.25, 0.25, 0.25]


def test_lt_gemini_transform_parses_json_array():
    class _FakeGeminiModel:
        def generate_content(self, prompt_text, generation_config=None):
            assert "Inputs" in prompt_text
            return types.SimpleNamespace(text='["x1", "x2"]')

    out = utils_mod.gemini_transform(
        texts=["a", "b"],
        params={
            "gemini_model": _FakeGeminiModel(),
            "model": "gemini-2.0-flash",
            "prompt": "Rewrite",
            "openai_params": {},
            "max_retries": 1,
            "ratelimit_sleep_time": 0,
            "retry_policy": {"max_retries": 1, "base_sleep_seconds": 0},
        },
    )

    assert out == ["x1", "x2"]


def test_lt_predict_rows_with_llm_resume_from_batch_checkpoint(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "llm_batch_checkpoint.jsonl"
    checkpoint_path.write_text(
        json.dumps(
            {
                "kind": "llm_classification_batch",
                "start": 0,
                "end": 3,
                "responses": ["Yes", "No", "Yes"],
                "provider": "openai",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(utils_mod.openai, "__version__", "1.3.0")
    monkeypatch.setattr(utils_mod.openai, "OpenAI", lambda **kwargs: object())

    def _should_not_call(*args, **kwargs):
        raise AssertionError("No API call expected when batch checkpoint fully covers rows")

    monkeypatch.setattr(utils_mod, "_openai_chat_completion_resilient", _should_not_call)

    out = utils_mod.predict_rows_with_llm(
        strings_col=["a", "b", "c"],
        model="gpt-4o-mini",
        openai_key="dummy",
        llm_provider="openai",
        openai_prompt="Classify",
        label_dict={"Yes": 1, "No": 0},
        checkpoint_path=str(checkpoint_path),
        resume=True,
        progress_bar=False,
        retry_policy={"max_retries": 1, "base_sleep_seconds": 0},
    )

    assert out == [1, 0, 1]


def test_lt_transform_column_resume_from_batch_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "transform_batch_checkpoint.jsonl"
    checkpoint_path.write_text(
        json.dumps(
            {
                "kind": "transform_column_batch",
                "start": 0,
                "end": 3,
                "values": ["old-a", "old-b", "old-c"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    df = pd.DataFrame({"text": ["a", "b", "c"]})

    def _should_not_call(batch, params):
        raise AssertionError("No transform call expected when batch checkpoint fully covers rows")

    out = utils_mod.transform_column(
        df,
        column="text",
        transform_fn=_should_not_call,
        fn_kwargs={},
        chunk_size=2,
        output_column="tx",
        progress_bar=False,
        checkpoint_path=str(checkpoint_path),
        resume=True,
    )

    assert out["tx"].tolist() == ["old-a", "old-b", "old-c"]


def test_lt_predict_rows_with_llm_checkpoint_window_larger_than_batch(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "llm_window_checkpoint.jsonl"

    monkeypatch.setattr(utils_mod.openai, "__version__", "1.3.0")
    monkeypatch.setattr(utils_mod.openai, "OpenAI", lambda **kwargs: object())

    def _fake_resilient(client, request_kwargs):
        payload = json.loads(request_kwargs["messages"][1]["content"])
        labels = ["Yes" for _ in payload]
        msg = types.SimpleNamespace(content=json.dumps(labels))
        choice = types.SimpleNamespace(message=msg)
        return types.SimpleNamespace(choices=[choice])

    monkeypatch.setattr(utils_mod, "_openai_chat_completion_resilient", _fake_resilient)

    out = utils_mod.predict_rows_with_llm(
        strings_col=["a", "b", "c", "d", "e"],
        model="gpt-4o-mini",
        openai_key="dummy",
        llm_provider="openai",
        openai_prompt="Classify",
        label_dict={"Yes": 1, "No": 0},
        checkpoint_path=str(checkpoint_path),
        batch_size=2,
        checkpoint_window_size=4,
        progress_bar=False,
        retry_policy={"max_retries": 1, "base_sleep_seconds": 0},
    )

    assert out == [1, 1, 1, 1, 1]

    records = [json.loads(line) for line in checkpoint_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    batch_records = [r for r in records if r.get("kind") == "llm_classification_batch"]
    assert len(batch_records) == 2
    assert len(batch_records[0]["indices"]) == 4
    assert len(batch_records[1]["indices"]) == 1


def test_lt_transform_column_checkpoint_window_larger_than_chunk(tmp_path):
    checkpoint_path = tmp_path / "transform_window_checkpoint.jsonl"
    df = pd.DataFrame({"text": ["a", "b", "c", "d", "e"]})

    def _transform_fn(batch, params):
        return [f"tx::{item}" for item in batch]

    out = utils_mod.transform_column(
        df,
        column="text",
        transform_fn=_transform_fn,
        fn_kwargs={},
        chunk_size=2,
        checkpoint_window_size=4,
        output_column="tx",
        progress_bar=False,
        checkpoint_path=str(checkpoint_path),
        resume=False,
    )

    assert out["tx"].tolist() == ["tx::a", "tx::b", "tx::c", "tx::d", "tx::e"]

    records = [json.loads(line) for line in checkpoint_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    batch_records = [r for r in records if r.get("kind") == "transform_column_batch"]
    assert len(batch_records) == 2
    assert len(batch_records[0]["indices"]) == 4
    assert len(batch_records[1]["indices"]) == 1


def test_lt_predict_rows_with_llm_uses_thread_workers(monkeypatch):
    monkeypatch.setattr(utils_mod.openai, "__version__", "1.3.0")
    monkeypatch.setattr(utils_mod.openai, "OpenAI", lambda **kwargs: object())

    captured = {"max_workers": None, "calls": 0}

    class _FakeExecutor:
        def __init__(self, max_workers):
            captured["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            fut = concurrent.futures.Future()
            fut.set_result(fn(*args, **kwargs))
            return fut

    def _fake_as_completed(futures):
        return futures

    def _fake_resilient(client, request_kwargs):
        captured["calls"] += 1
        payload = json.loads(request_kwargs["messages"][1]["content"])
        labels = ["Yes" for _ in payload]
        msg = types.SimpleNamespace(content=json.dumps(labels))
        choice = types.SimpleNamespace(message=msg)
        return types.SimpleNamespace(choices=[choice])

    monkeypatch.setattr(utils_mod.concurrent.futures, "ThreadPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(utils_mod.concurrent.futures, "as_completed", _fake_as_completed)
    monkeypatch.setattr(utils_mod, "_openai_chat_completion_resilient", _fake_resilient)

    out = utils_mod.predict_rows_with_llm(
        strings_col=["a", "b", "c", "d"],
        model="gpt-4o-mini",
        openai_key="dummy",
        llm_provider="openai",
        openai_prompt="Classify",
        label_dict={"Yes": 1, "No": 0},
        batch_size=1,
        workers=3,
        progress_bar=False,
        retry_policy={"max_retries": 1, "base_sleep_seconds": 0},
    )

    assert out == [1, 1, 1, 1]
    assert captured["max_workers"] == 3
    assert captured["calls"] == 4


def test_lt_transform_column_uses_thread_workers(monkeypatch):
    captured = {"max_workers": None}

    class _FakeExecutor:
        def __init__(self, max_workers):
            captured["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            fut = concurrent.futures.Future()
            fut.set_result(fn(*args, **kwargs))
            return fut

    def _fake_as_completed(futures):
        return futures

    monkeypatch.setattr(utils_mod.concurrent.futures, "ThreadPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(utils_mod.concurrent.futures, "as_completed", _fake_as_completed)

    df = pd.DataFrame({"text": ["a", "b", "c", "d"]})

    out = utils_mod.transform_column(
        df,
        column="text",
        transform_fn=lambda batch, params: [f"tx::{x}" for x in batch],
        fn_kwargs={},
        chunk_size=1,
        workers=4,
        output_column="tx",
        progress_bar=False,
    )

    assert out["tx"].tolist() == ["tx::a", "tx::b", "tx::c", "tx::d"]
    assert captured["max_workers"] == 4