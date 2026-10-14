import os
import importlib.machinery
import json
import sys
import types
import time
import threading

import openai
import pandas as pd
import pytest

import linktransformer.infer as infer_mod


@pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="OpenAI API key not found in environment")
def test_lt_merge_k_judge_openai_live():
    df1 = pd.DataFrame(
        {
            "CompanyName": ["Tech Corporation", "InfoTech Solutions"],
            "Country": ["USA", "USA"],
        }
    )
    df2 = pd.DataFrame(
        {
            "CompanyName": ["Tech Corp", "InfoTech Soln"],
            "Country": ["USA", "USA"],
        }
    )

    out = infer_mod.merge_k_judge(
        df1=df1,
        df2=df2,
        on=["CompanyName", "Country"],
        model="text-embedding-3-small",
        openai_key=os.getenv("OPENAI_API_KEY"),
        k=1,
        llm_provider="openai",
        judge_llm_model="gpt-4o-mini",
        confidence_threshold=0.0,
    )

    display_cols = [
        "CompanyName_x",
        "CompanyName_y",
        "Country_x",
        "Country_y",
        "score",
        "llm_is_match",
        "llm_confidence",
    ]
    print("\n[OpenAI] merge_k_judge output:")
    print(out[display_cols])
    print(f"[OpenAI] rows={len(out)}, mean_conf={out['llm_confidence'].mean():.3f}")

    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(df1)
    assert "llm_is_match" in out.columns
    assert "llm_confidence" in out.columns
    assert out["llm_confidence"].between(0, 1).all()


@pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="OpenAI API key not found in environment")
def test_lt_merge_k_judge_sbert_retrieval_openai_llm_live():
    df1 = pd.DataFrame(
        {
            "CompanyName": ["Tech Corporation", "InfoTech Solutions"],
            "Country": ["USA", "USA"],
        }
    )
    df2 = pd.DataFrame(
        {
            "CompanyName": ["Tech Corp", "InfoTech Soln"],
            "Country": ["USA", "USA"],
        }
    )

    out = infer_mod.merge_k_judge(
        df1=df1,
        df2=df2,
        on=["CompanyName", "Country"],
        model="sentence-transformers/all-MiniLM-L6-v2",
        openai_key=os.getenv("OPENAI_API_KEY"),
        k=1,
        llm_provider="openai",
        judge_llm_model="gpt-4o-mini",
        confidence_threshold=0.0,
    )

    display_cols = [
        "CompanyName_x",
        "CompanyName_y",
        "Country_x",
        "Country_y",
        "score",
        "llm_is_match",
        "llm_confidence",
    ]
    print("\n[SBERT + OpenAI] merge_k_judge output:")
    print(out[display_cols])
    print(f"[SBERT + OpenAI] rows={len(out)}, mean_conf={out['llm_confidence'].mean():.3f}")

    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(df1)
    assert "llm_is_match" in out.columns
    assert "llm_confidence" in out.columns
    assert out["llm_confidence"].between(0, 1).all()


@pytest.mark.skipif("GEMINI_API_KEY" not in os.environ, reason="Gemini API key not found in environment")
def test_lt_merge_k_judge_gemini_live():
    pytest.importorskip("google.genai")

    df1 = pd.DataFrame(
        {
            "CompanyName": ["Tech Corporation", "AlphaSoft Systems"],
            "Country": ["USA", "Canada"],
        }
    )
    df2 = pd.DataFrame(
        {
            "CompanyName": ["Tech Corp", "AlphaSoft"],
            "Country": ["USA", "Canada"],
        }
    )

    out = infer_mod.merge_k_judge(
        df1=df1,
        df2=df2,
        on=["CompanyName", "Country"],
        model="gemini-embedding-001",
        gemini_key=os.getenv("GEMINI_API_KEY"),
        k=1,
        llm_provider="gemini",
        judge_llm_model="gemini-3.5-flash-lite",
        confidence_threshold=0.0,
    )

    display_cols = [
        "CompanyName_x",
        "CompanyName_y",
        "Country_x",
        "Country_y",
        "score",
        "llm_is_match",
        "llm_confidence",
    ]
    print("\n[Gemini] merge_k_judge output:")
    print(out[display_cols])
    print(f"[Gemini] rows={len(out)}, mean_conf={out['llm_confidence'].mean():.3f}")

    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(df1)
    assert "llm_is_match" in out.columns
    assert "llm_confidence" in out.columns
    assert out["llm_confidence"].between(0, 1).all()


@pytest.mark.skipif("GEMINI_API_KEY" not in os.environ, reason="Gemini API key not found in environment")
def test_lt_merge_k_judge_sbert_retrieval_gemini_judge_live():
    pytest.importorskip("google.genai")

    df1 = pd.DataFrame(
        {
            "CompanyName": ["Tech Corporation", "AlphaSoft Systems"],
            "Country": ["USA", "Canada"],
        }
    )
    df2 = pd.DataFrame(
        {
            "CompanyName": ["Tech Corp", "AlphaSoft"],
            "Country": ["USA", "Canada"],
        }
    )

    out = infer_mod.merge_k_judge(
        df1=df1,
        df2=df2,
        on=["CompanyName", "Country"],
        model="all-MiniLM-L6-v2",
        knn_sbert_model="sentence-transformers/all-MiniLM-L6-v2",
        gemini_key=os.getenv("GEMINI_API_KEY"),
        k=1,
        llm_provider="gemini",
        judge_llm_model="gemini-3.5-flash-lite",
        confidence_threshold=0.0,
    )

    display_cols = [
        "CompanyName_x",
        "CompanyName_y",
        "Country_x",
        "Country_y",
        "score",
        "llm_is_match",
        "llm_confidence",
    ]
    print("\n[SBERT + Gemini judge] merge_k_judge output:")
    print(out[display_cols])
    print(f"[SBERT + Gemini judge] rows={len(out)}, mean_conf={out['llm_confidence'].mean():.3f}")

    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(df1)
    assert "llm_is_match" in out.columns
    assert "llm_confidence" in out.columns
    assert out["llm_confidence"].between(0, 1).all()


@pytest.mark.skipif(
    ("OPENAI_API_KEY" not in os.environ) or ("GEMINI_API_KEY" not in os.environ),
    reason="Both OpenAI and Gemini API keys are required",
)
def test_lt_merge_k_judge_openai_embedding_gemini_judge_live():
    pytest.importorskip("google.genai")

    df1 = pd.DataFrame(
        {
            "CompanyName": ["Tech Corporation", "InfoTech Solutions"],
            "Country": ["USA", "USA"],
        }
    )
    df2 = pd.DataFrame(
        {
            "CompanyName": ["Tech Corp", "InfoTech Soln"],
            "Country": ["USA", "USA"],
        }
    )

    out = infer_mod.merge_k_judge(
        df1=df1,
        df2=df2,
        on=["CompanyName", "Country"],
        model="all-MiniLM-L6-v2",
        knn_api_model="text-embedding-3-small",
        openai_key=os.getenv("OPENAI_API_KEY"),
        gemini_key=os.getenv("GEMINI_API_KEY"),
        k=1,
        llm_provider="gemini",
        judge_llm_model="gemini-3.5-flash-lite",
        confidence_threshold=0.0,
    )

    display_cols = [
        "CompanyName_x",
        "CompanyName_y",
        "Country_x",
        "Country_y",
        "score",
        "llm_is_match",
        "llm_confidence",
    ]
    print("\n[OpenAI embedding + Gemini judge] merge_k_judge output:")
    print(out[display_cols])
    print(f"[OpenAI embedding + Gemini judge] rows={len(out)}, mean_conf={out['llm_confidence'].mean():.3f}")

    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(df1)
    assert "llm_is_match" in out.columns
    assert "llm_confidence" in out.columns
    assert out["llm_confidence"].between(0, 1).all()


def test_lt_merge_k_judge_uses_knn_sbert_model(monkeypatch):
    captured = {}

    def fake_merge_knn(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "CompanyName_x": ["Tech Corporation"],
                "CompanyName_y": ["Tech Corp"],
                "Country_x": ["USA"],
                "Country_y": ["USA"],
                "score": [0.95],
            }
        )

    monkeypatch.setattr(infer_mod, "merge_knn", fake_merge_knn)

    out = infer_mod.merge_k_judge(
        df1=pd.DataFrame({"CompanyName": ["Tech Corporation"], "Country": ["USA"]}),
        df2=pd.DataFrame({"CompanyName": ["Tech Corp"], "Country": ["USA"]}),
        on=["CompanyName", "Country"],
        model="all-MiniLM-L6-v2",
        knn_sbert_model="sentence-transformers/all-MiniLM-L6-v2",
        openai_key="dummy",
        llm_provider="openai",
        judge_llm_model="gpt-4o-mini",
        max_retries=0,
    )
    

    assert captured["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert captured["openai_key"] is None
    assert captured["gemini_key"] is None
    assert "llm_confidence" in out.columns


def test_lt_merge_k_judge_warns_when_only_model_specified(monkeypatch):
    def fake_merge_knn(**kwargs):
        return pd.DataFrame(
            {
                "CompanyName_x": ["Tech Corporation"],
                "CompanyName_y": ["Tech Corp"],
                "Country_x": ["USA"],
                "Country_y": ["USA"],
                "score": [0.95],
            }
        )

    monkeypatch.setattr(infer_mod, "merge_knn", fake_merge_knn)

    with pytest.warns(UserWarning, match="Using `model` as shared default"):
        out = infer_mod.merge_k_judge(
            df1=pd.DataFrame({"CompanyName": ["Tech Corporation"], "Country": ["USA"]}),
            df2=pd.DataFrame({"CompanyName": ["Tech Corp"], "Country": ["USA"]}),
            on=["CompanyName", "Country"],
            model="all-MiniLM-L6-v2",
            openai_key="dummy",
            llm_provider="openai",
            judge_llm_model="gpt-4o-mini",
            max_retries=0,
        )

    assert isinstance(out, pd.DataFrame)
    assert "llm_confidence" in out.columns


def test_lt_merge_k_judge_allows_judge_on_override(monkeypatch):
    monkeypatch.setattr(
        infer_mod,
        "merge_knn",
        lambda **kwargs: pd.DataFrame(
            {
                "CompanyName_x": ["Tech Corporation"],
                "CompanyName_y": ["Tech Corp"],
                "Address_x": ["123 Main St"],
                "Address_y": ["123 Main Street"],
                "score": [0.95],
            }
        ),
    )

    captured = {}

    class _OkCompletions:
        def create(self, **kwargs):
            captured["messages"] = kwargs.get("messages")

            class _Msg:
                content = '{"is_match": 1, "confidence": 0.9}'

            class _Choice:
                message = _Msg()

            class _Resp:
                choices = [_Choice()]

            return _Resp()

    class _OkChat:
        def __init__(self):
            self.completions = _OkCompletions()

    class _OkOpenAIClient:
        def __init__(self, api_key, timeout):
            self.chat = _OkChat()

    monkeypatch.setattr(infer_mod.openai, "OpenAI", _OkOpenAIClient)

    out = infer_mod.merge_k_judge(
        df1=pd.DataFrame({"CompanyName": ["Tech Corporation"], "Address": ["123 Main St"]}),
        df2=pd.DataFrame({"CompanyName": ["Tech Corp"], "Address": ["123 Main Street"]}),
        on=["CompanyName"],
        judge_on=["Address"],
        model="all-MiniLM-L6-v2",
        openai_key="dummy",
        llm_provider="openai",
        judge_llm_model="gpt-4o-mini",
        max_retries=1,
    )

    # The LLM payload should only include the judge columns, not the retrieval columns.
    user_msg = captured["messages"][1]["content"]
    payload = json.loads(user_msg)
    assert set(payload["left"].keys()) == {"Address"}
    assert set(payload["right"].keys()) == {"Address"}
    assert "llm_confidence" in out.columns


def test_lt_merge_k_judge_allows_asymmetric_judge_columns(monkeypatch):
    monkeypatch.setattr(
        infer_mod,
        "merge_knn",
        lambda **kwargs: pd.DataFrame(
            {
                "CompanyName_x": ["Tech Corporation"],
                "CompanyName_y": ["Tech Corp"],
                "Address": ["123 Main St"],
                "Addr": ["123 Main Street"],
                "score": [0.95],
            }
        ),
    )

    captured = {}

    class _OkCompletions:
        def create(self, **kwargs):
            captured["messages"] = kwargs.get("messages")

            class _Msg:
                content = '{"is_match": 1, "confidence": 0.9}'

            class _Choice:
                message = _Msg()

            class _Resp:
                choices = [_Choice()]

            return _Resp()

    class _OkChat:
        def __init__(self):
            self.completions = _OkCompletions()

    class _OkOpenAIClient:
        def __init__(self, api_key, timeout):
            self.chat = _OkChat()

    monkeypatch.setattr(infer_mod.openai, "OpenAI", _OkOpenAIClient)

    out = infer_mod.merge_k_judge(
        df1=pd.DataFrame({"CompanyName": ["Tech Corporation"], "Address": ["123 Main St"]}),
        df2=pd.DataFrame({"CompanyName": ["Tech Corp"], "Addr": ["123 Main Street"]}),
        on=["CompanyName"],
        judge_left_on=["Address"],
        judge_right_on=["Addr"],
        model="all-MiniLM-L6-v2",
        openai_key="dummy",
        llm_provider="openai",
        judge_llm_model="gpt-4o-mini",
        max_retries=1,
    )

    user_msg = captured["messages"][1]["content"]
    payload = json.loads(user_msg)
    assert set(payload["left"].keys()) == {"Address"}
    assert set(payload["right"].keys()) == {"Addr"}
    assert "llm_confidence" in out.columns


def test_lt_merge_k_judge_skips_judge_above_upper_threshold(monkeypatch):
    monkeypatch.setattr(
        infer_mod,
        "merge_knn",
        lambda **kwargs: pd.DataFrame(
            {
                "CompanyName_x": ["Tech Corporation"],
                "CompanyName_y": ["Tech Corp"],
                "score": [0.95],
            }
        ),
    )

    class _NoJudgeCompletions:
        def create(self, **kwargs):
            raise AssertionError("LLM judge should have been skipped for high-sim rows")

    class _NoJudgeChat:
        def __init__(self):
            self.completions = _NoJudgeCompletions()

    class _NoJudgeOpenAIClient:
        def __init__(self, api_key, timeout):
            self.chat = _NoJudgeChat()

    monkeypatch.setattr(infer_mod.openai, "OpenAI", _NoJudgeOpenAIClient)

    out = infer_mod.merge_k_judge(
        df1=pd.DataFrame({"CompanyName": ["Tech Corporation"]}),
        df2=pd.DataFrame({"CompanyName": ["Tech Corp"]}),
        on=["CompanyName"],
        model="all-MiniLM-L6-v2",
        openai_key="dummy",
        llm_provider="openai",
        judge_llm_model="gpt-4o-mini",
        judge_skip_upper_thresh=0.9,
        max_retries=1,
    )

    assert out.loc[0, "llm_is_match"] == 99
    assert out.loc[0, "llm_confidence"] == 99.0
    assert out.loc[0, "llm_raw_response"] == "99"


def test_lt_merge_k_judge_skips_judge_below_lower_threshold(monkeypatch):
    monkeypatch.setattr(
        infer_mod,
        "merge_knn",
        lambda **kwargs: pd.DataFrame(
            {
                "CompanyName_x": ["Tech Corporation"],
                "CompanyName_y": ["Tech Corp"],
                "score": [0.05],
            }
        ),
    )

    class _NoJudgeCompletions:
        def create(self, **kwargs):
            raise AssertionError("LLM judge should have been skipped for low-sim rows")

    class _NoJudgeChat:
        def __init__(self):
            self.completions = _NoJudgeCompletions()

    class _NoJudgeOpenAIClient:
        def __init__(self, api_key, timeout):
            self.chat = _NoJudgeChat()

    monkeypatch.setattr(infer_mod.openai, "OpenAI", _NoJudgeOpenAIClient)

    out = infer_mod.merge_k_judge(
        df1=pd.DataFrame({"CompanyName": ["Tech Corporation"]}),
        df2=pd.DataFrame({"CompanyName": ["Tech Corp"]}),
        on=["CompanyName"],
        model="all-MiniLM-L6-v2",
        openai_key="dummy",
        llm_provider="openai",
        judge_llm_model="gpt-4o-mini",
        judge_skip_lower_thresh=0.1,
        max_retries=1,
    )

    assert out.loc[0, "llm_is_match"] == 99
    assert out.loc[0, "llm_confidence"] == 99.0
    assert out.loc[0, "llm_raw_response"] == "99"


def test_lt_judge_merge_k_out_skips_already_judged_rows(monkeypatch):
    candidates = pd.DataFrame(
        {
            "CompanyName_x": ["A", "B"],
            "CompanyName_y": ["A2", "B2"],
            "score": [0.9, 0.8],
            "llm_is_match": [1, None],
            "llm_confidence": [0.91, None],
            "llm_raw_response": ['{"is_match":1,"confidence":0.91}', None],
        }
    )

    calls = {"n": 0}

    class _FakeCompletions:
        def create(self, **kwargs):
            calls["n"] += 1
            msg = types.SimpleNamespace(content='{"is_match": 0, "confidence": 0.22}')
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeCompletions()

    class _FakeOpenAIClient:
        def __init__(self, api_key, timeout):
            self.chat = _FakeChat()

    monkeypatch.setattr(infer_mod.openai, "OpenAI", _FakeOpenAIClient)

    out = infer_mod.judge_merge_k_out(
        candidates_df=candidates,
        on="CompanyName",
        openai_key="dummy",
        llm_provider="openai",
        judge_llm_model="gpt-4o-mini",
        max_retries=1,
        progress_bar=False,
    )

    assert calls["n"] == 1
    assert out.loc[0, "llm_is_match"] == 1
    assert out.loc[0, "llm_confidence"] == 0.91
    assert out.loc[1, "llm_is_match"] == 0
    assert out.loc[1, "llm_confidence"] == 0.22


def test_lt_judge_merge_k_out_honors_judge_workers(monkeypatch):
    candidates = pd.DataFrame(
        {
            "CompanyName_x": [f"A{i}" for i in range(6)],
            "CompanyName_y": [f"B{i}" for i in range(6)],
            "score": [0.8] * 6,
        }
    )

    state = {"active": 0, "max_active": 0}
    state_lock = threading.Lock()

    class _FakeCompletions:
        def create(self, **kwargs):
            with state_lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.03)
            with state_lock:
                state["active"] -= 1
            msg = types.SimpleNamespace(content='{"is_match": 1, "confidence": 0.8}')
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeCompletions()

    class _FakeOpenAIClient:
        def __init__(self, api_key, timeout):
            self.chat = _FakeChat()

    monkeypatch.setattr(infer_mod.openai, "OpenAI", _FakeOpenAIClient)

    out = infer_mod.judge_merge_k_out(
        candidates_df=candidates,
        on="CompanyName",
        openai_key="dummy",
        llm_provider="openai",
        judge_llm_model="gpt-4o-mini",
        max_retries=1,
        progress_bar=False,
        judge_workers=3,
    )

    assert state["max_active"] <= 3
    assert state["max_active"] >= 2
    assert (out["llm_is_match"] == 1).all()


def test_lt_merge_k_judge_errors_when_judge_model_missing(monkeypatch):
    monkeypatch.setattr(
        infer_mod,
        "merge_knn",
        lambda **kwargs: pd.DataFrame({"CompanyName_x": ["a"], "CompanyName_y": ["b"], "score": [0.9]}),
    )

    with pytest.raises(ValueError, match="requires `judge_llm_model`"):
        infer_mod.merge_k_judge(
            df1=pd.DataFrame({"CompanyName": ["a"]}),
            df2=pd.DataFrame({"CompanyName": ["b"]}),
            on="CompanyName",
            model="all-MiniLM-L6-v2",
            openai_key="dummy",
            llm_provider="openai",
        )


def test_lt_merge_k_judge_errors_on_openai_judge_failure(monkeypatch):
    knn_out = pd.DataFrame(
        {
            "CompanyName_x": ["Tech Corporation"],
            "CompanyName_y": ["Tech Corp"],
            "Country_x": ["USA"],
            "Country_y": ["USA"],
            "score": [0.95],
        }
    )

    monkeypatch.setattr(infer_mod, "merge_knn", lambda **kwargs: knn_out.copy())

    class _FailingCompletions:
        def create(self, **kwargs):
            raise RuntimeError("invalid openai key or model")

    class _FailingChat:
        def __init__(self):
            self.completions = _FailingCompletions()

    class _FailingOpenAIClient:
        def __init__(self, api_key, timeout):
            self.chat = _FailingChat()

    monkeypatch.setattr(openai, "OpenAI", _FailingOpenAIClient)

    with pytest.raises(RuntimeError, match="Use merge_knn"):
        infer_mod.merge_k_judge(
            df1=pd.DataFrame({"CompanyName": ["Tech Corporation"], "Country": ["USA"]}),
            df2=pd.DataFrame({"CompanyName": ["Tech Corp"], "Country": ["USA"]}),
            on=["CompanyName", "Country"],
            model="all-MiniLM-L6-v2",
            knn_sbert_model="sentence-transformers/all-MiniLM-L6-v2",
            openai_key="bad-key",
            llm_provider="openai",
            judge_llm_model="bad-model",
            max_retries=1,
        )


def test_lt_merge_k_judge_errors_on_gemini_judge_failure(monkeypatch):
    knn_out = pd.DataFrame(
        {
            "CompanyName_x": ["AlphaSoft Systems"],
            "CompanyName_y": ["AlphaSoft"],
            "Country_x": ["Canada"],
            "Country_y": ["Canada"],
            "score": [0.94],
        }
    )

    monkeypatch.setattr(infer_mod, "merge_knn", lambda **kwargs: knn_out.copy())

    class _FailingModels:
        def generate_content(self, model, contents, config):
            raise RuntimeError("invalid gemini key or model")

    class _FailingGeminiClient:
        def __init__(self, api_key, http_options=None):
            self.models = _FailingModels()

    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = _FailingGeminiClient
    fake_genai.types = types.SimpleNamespace(
        GenerateContentConfig=lambda **kwargs: kwargs,
        HttpOptions=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    with pytest.raises(RuntimeError, match="Use merge_knn"):
        infer_mod.merge_k_judge(
            df1=pd.DataFrame({"CompanyName": ["AlphaSoft Systems"], "Country": ["Canada"]}),
            df2=pd.DataFrame({"CompanyName": ["AlphaSoft"], "Country": ["Canada"]}),
            on=["CompanyName", "Country"],
            model="all-MiniLM-L6-v2",
            knn_sbert_model="sentence-transformers/all-MiniLM-L6-v2",
            gemini_key="bad-key",
            llm_provider="gemini",
            judge_llm_model="bad-model",
            max_retries=1,
        )


def test_lt_coerce_llm_match_and_confidence_dict_and_bounds():
    is_match, confidence = infer_mod._coerce_llm_match_and_confidence(
        '{"is_match": true, "confidence": 1.7}'
    )

    assert is_match == 1
    assert confidence == 1.0


def test_lt_coerce_llm_match_and_confidence_fallback_text():
    is_match, confidence = infer_mod._coerce_llm_match_and_confidence("Yes, this is a match. confidence: 0.75")

    assert is_match == 1
    assert confidence == 0.75


@pytest.mark.parametrize(
    "reply, expected",
    [
        # some models (often Gemini) wrap the JSON in a code fence; "is_match" must not read as the word "match"
        ('```json\n{"is_match": 0, "confidence": 0.9}\n```', (0, 0.9)),
        ('```json\n{"is_match": 1, "confidence": 0.8}\n```', (1, 0.8)),
        ('```\n{"is_match": 0, "confidence": 0.6}\n```', (0, 0.6)),
        ('Here is my answer: {"is_match": 0, "confidence": 0.7}', (0, 0.7)),
        ('{"is_match": 0, "confid', (0, 0.0)),  # cut off by max_tokens
        ("No, this is not a match. confidence: 0.85", (0, 0.85)),
        ("", (0, 0.0)),
    ],
)
def test_lt_coerce_llm_match_and_confidence_fenced_and_free_text(reply, expected):
    assert infer_mod._coerce_llm_match_and_confidence(reply) == expected


def test_lt_judge_merge_k_out_confidence_threshold_drops_low_score_skips(monkeypatch):
    candidates = pd.DataFrame(
        {
            "CompanyName_x": ["A", "B", "C", "D"],
            "CompanyName_y": ["A2", "B2", "C2", "D2"],
            "score": [0.99, 0.70, 0.60, 0.05],
        }
    )
    replies = {"B": '{"is_match": 1, "confidence": 0.95}', "C": '{"is_match": 0, "confidence": 0.40}'}

    class _FakeCompletions:
        def create(self, **kwargs):
            left = json.loads(kwargs["messages"][1]["content"])["left"]["CompanyName"]
            msg = types.SimpleNamespace(content=replies[left])
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class _FakeOpenAIClient:
        def __init__(self, api_key, timeout):
            self.chat = types.SimpleNamespace(completions=_FakeCompletions())

    monkeypatch.setattr(infer_mod.openai, "OpenAI", _FakeOpenAIClient)

    out = infer_mod.judge_merge_k_out(
        candidates_df=candidates,
        on="CompanyName",
        openai_key="dummy",
        llm_provider="openai",
        judge_llm_model="gpt-4o-mini",
        judge_skip_lower_thresh=0.1,
        judge_skip_upper_thresh=0.95,
        confidence_threshold=0.5,
        max_retries=1,
        progress_bar=False,
    )

    # A: skipped for a high score (kept); B: judged, confident (kept); C: judged, not confident (dropped);
    # D: skipped for a low score (dropped, although its sentinel confidence 99 is above the threshold)
    assert out["CompanyName_x"].tolist() == ["A", "B"]
    assert out["llm_is_match"].tolist() == [99, 1]


@pytest.mark.parametrize(
    "response_text, expected",
    [
        # Gemini sometimes wraps its JSON in a Markdown code fence.
        ('```json\n{"is_match": 0, "confidence": 0.95}\n```', (0, 0.95)),
        ('```json {"is_match": 0, "confidence": 0.9} ```', (0, 0.9)),
        ('```\n{"is_match": 1, "confidence": 0.8}\n```', (1, 0.8)),
        ('Here is my answer: {"is_match": 0, "confidence": 0.7}', (0, 0.7)),
        # Truncated JSON (closing brace or fence cut off by the token limit).
        ('```json\n{"is_match": 0, "confidence": 0.85', (0, 0.85)),
        ('{"is_match": true, "confidence": 0.6', (1, 0.6)),
        ('{"is_match": "no"', (0, 0.0)),
    ],
)
def test_lt_coerce_llm_match_and_confidence_fenced_and_truncated_json(response_text, expected):
    assert infer_mod._coerce_llm_match_and_confidence(response_text) == expected


@pytest.mark.parametrize(
    "response_text, expected_match",
    [
        ("No, these are different companies.", 0),
        ("Not a match.", 0),
        ("false", 0),
        ("These are a match.", 1),
        ("I cannot tell.", 0),
    ],
)
def test_lt_coerce_llm_match_and_confidence_free_text_answers(response_text, expected_match):
    is_match, _ = infer_mod._coerce_llm_match_and_confidence(response_text)

    assert is_match == expected_match


def test_lt_judge_merge_k_out_gemini_fenced_rejections_are_not_matches(monkeypatch):
    candidates = pd.DataFrame(
        {
            "CompanyName_x": ["Nagaraja m b", "Tech Corporation"],
            "CompanyName_y": ["Nagarjuna Cement", "Tech Corp"],
            "score": [0.71, 0.93],
        }
    )
    responses = {
        "Nagaraja m b": '```json\n{"is_match": 0, "confidence": 0.97}\n```',
        "Tech Corporation": '```json\n{"is_match": 1, "confidence": 0.9}\n```',
    }

    configs = []

    class _FencedModels:
        def generate_content(self, model, contents, config):
            configs.append(config)
            left_name = next(name for name in responses if name in contents)
            return types.SimpleNamespace(text=responses[left_name])

    class _FencedGeminiClient:
        def __init__(self, api_key, http_options=None):
            self.models = _FencedModels()

    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = _FencedGeminiClient
    fake_genai.types = types.SimpleNamespace(
        GenerateContentConfig=lambda **kwargs: kwargs,
        HttpOptions=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    out = infer_mod.judge_merge_k_out(
        candidates_df=candidates,
        on="CompanyName",
        gemini_key="dummy",
        llm_provider="gemini",
        judge_llm_model="gemini-2.5-flash",
        max_retries=1,
        progress_bar=False,
    )

    assert out.loc[0, "llm_is_match"] == 0
    assert out.loc[0, "llm_confidence"] == 0.97
    assert out.loc[1, "llm_is_match"] == 1
    assert out.loc[1, "llm_confidence"] == 0.9
    assert len(configs) == 2
    assert all(config["response_mime_type"] == "application/json" for config in configs)


def test_lt_resolve_knn_retrieval_config_mutual_exclusion_error():
    with pytest.raises(ValueError, match="Specify only one"):
        infer_mod._resolve_knn_retrieval_config(
            model="all-MiniLM-L6-v2",
            knn_sbert_model="sentence-transformers/all-MiniLM-L6-v2",
            knn_api_model="text-embedding-3-small",
            openai_key="dummy",
            gemini_key=None,
        )


def test_lt_resolve_knn_api_model_prefers_gemini_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    model = infer_mod._resolve_knn_api_model(
        knn_api_model="auto",
        openai_key=None,
        gemini_key=None,
    )
    assert model == "gemini-embedding-001"


def test_lt_resolve_knn_retrieval_config_openai_api_keys():
    retrieval_model, retrieval_openai_key, retrieval_gemini_key = infer_mod._resolve_knn_retrieval_config(
        model="all-MiniLM-L6-v2",
        knn_sbert_model=None,
        knn_api_model="text-embedding-3-small",
        openai_key="openai-key",
        gemini_key="gemini-key",
    )

    assert retrieval_model == "text-embedding-3-small"
    assert retrieval_openai_key == "openai-key"
    assert retrieval_gemini_key is None


def test_lt_merge_k_judge_resume_from_checkpoint(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "judge_checkpoint.jsonl"
    checkpoint_path.write_text(
        json.dumps(
            {
                "kind": "merge_k_judge_row",
                "index": 0,
                "llm_is_match": 1,
                "llm_confidence": 0.91,
                "llm_raw_response": '{"is_match":1,"confidence":0.91}',
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        infer_mod,
        "merge_knn",
        lambda **kwargs: pd.DataFrame(
            {
                "CompanyName_x": ["A", "B"],
                "CompanyName_y": ["A2", "B2"],
                "score": [0.9, 0.8],
            }
        ),
    )

    calls = {"n": 0}

    class _FakeCompletions:
        def create(self, **kwargs):
            calls["n"] += 1
            msg = types.SimpleNamespace(content='{"is_match": 0, "confidence": 0.22}')
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeCompletions()

    class _FakeOpenAIClient:
        def __init__(self, api_key, timeout):
            self.chat = _FakeChat()

    monkeypatch.setattr(infer_mod.openai, "OpenAI", _FakeOpenAIClient)

    out = infer_mod.merge_k_judge(
        df1=pd.DataFrame({"CompanyName": ["A", "B"]}),
        df2=pd.DataFrame({"CompanyName": ["A2", "B2"]}),
        on="CompanyName",
        model="all-MiniLM-L6-v2",
        openai_key="dummy",
        llm_provider="openai",
        judge_llm_model="gpt-4o-mini",
        max_retries=1,
        checkpoint_path=str(checkpoint_path),
        resume=True,
        retry_policy={"max_retries": 1, "base_sleep_seconds": 0},
    )

    assert len(out) == 2
    assert out.loc[0, "llm_is_match"] == 1
    assert out.loc[0, "llm_confidence"] == 0.91
    assert out.loc[1, "llm_is_match"] == 0
    assert out.loc[1, "llm_confidence"] == 0.22
    assert calls["n"] == 1


def test_lt_merge_k_judge_logs_failure(monkeypatch, tmp_path):
    failure_log_path = tmp_path / "judge_failures.jsonl"

    monkeypatch.setattr(
        infer_mod,
        "merge_knn",
        lambda **kwargs: pd.DataFrame({"CompanyName_x": ["A"], "CompanyName_y": ["A2"], "score": [0.9]}),
    )

    class _FailingCompletions:
        def create(self, **kwargs):
            raise RuntimeError("judge api failed")

    class _FailingChat:
        def __init__(self):
            self.completions = _FailingCompletions()

    class _FailingOpenAIClient:
        def __init__(self, api_key, timeout):
            self.chat = _FailingChat()

    monkeypatch.setattr(infer_mod.openai, "OpenAI", _FailingOpenAIClient)

    with pytest.raises(RuntimeError, match="Use merge_knn"):
        infer_mod.merge_k_judge(
            df1=pd.DataFrame({"CompanyName": ["A"]}),
            df2=pd.DataFrame({"CompanyName": ["A2"]}),
            on="CompanyName",
            model="all-MiniLM-L6-v2",
            openai_key="dummy",
            llm_provider="openai",
            judge_llm_model="gpt-4o-mini",
            max_retries=1,
            failure_log_path=str(failure_log_path),
            retry_policy={"max_retries": 1, "base_sleep_seconds": 0},
        )

    payload = json.loads(failure_log_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert payload["function"] == "merge_k_judge"
    assert payload["provider"] == "openai"

