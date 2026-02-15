import os
import importlib.machinery
import sys
import types

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
    pytest.importorskip("google.generativeai")

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
        judge_llm_model="gemini-2.0-flash",
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
    pytest.importorskip("google.generativeai")

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
        judge_llm_model="gemini-2.0-flash",
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
    pytest.importorskip("google.generativeai")

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
        judge_llm_model="gemini-2.0-flash",
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

    monkeypatch.setattr(infer_mod.openai, "OpenAI", _FailingOpenAIClient)

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

    class _FailingGeminiModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_content(self, prompt):
            raise RuntimeError("invalid gemini key or model")

    fake_genai = types.ModuleType("google.generativeai")
    fake_genai.configure = lambda api_key: None
    fake_genai.GenerativeModel = _FailingGeminiModel
    fake_genai.__spec__ = importlib.machinery.ModuleSpec(name="google.generativeai", loader=None)

    google_pkg = types.ModuleType("google")
    google_pkg.__spec__ = importlib.machinery.ModuleSpec(name="google", loader=None)
    google_pkg.generativeai = fake_genai

    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)

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

