import sys
import types

import numpy as np
import openai
import pytest

import linktransformer.utils as utils_mod


def test_lt_gemini_key_takes_priority_over_openai_key(monkeypatch):
    expected = np.array([[0.2, 0.4]], dtype=np.float32)

    def fake_gemini(strings, model, api_key, return_numpy=True):
        assert strings == ["alpha"]
        assert model == "gemini-embedding-001"
        assert api_key == "gemini-key"
        assert return_numpy is True
        return expected

    monkeypatch.setattr(utils_mod, "infer_embeddings_with_gemini", fake_gemini)

    out = utils_mod.infer_embeddings(
        strings=["alpha"],
        model="gemini-embedding-001",
        gemini_key="gemini-key",
        openai_key="openai-key",
        return_numpy=True,
    )

    assert np.allclose(out, expected)


def test_lt_gemini_env_fallback_when_no_explicit_key(monkeypatch):
    expected = np.array([[0.1, 0.3]], dtype=np.float32)

    def fake_gemini(strings, model, api_key, return_numpy=True):
        assert api_key == "env-gemini-key"
        return expected

    monkeypatch.setenv("GEMINI_API_KEY", "env-gemini-key")
    monkeypatch.setattr(utils_mod, "infer_embeddings_with_gemini", fake_gemini)

    out = utils_mod.infer_embeddings(
        strings=["beta"],
        model="gemini-embedding-001",
        gemini_key=None,
        openai_key=None,
        return_numpy=True,
    )

    assert np.allclose(out, expected)


def test_lt_gemini_legacy_openai_key_fallback(monkeypatch):
    expected = np.array([[0.7, 0.8]], dtype=np.float32)

    def fake_gemini(strings, model, api_key, return_numpy=True):
        assert api_key == "legacy-openai-key"
        return expected

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(utils_mod, "infer_embeddings_with_gemini", fake_gemini)

    out = utils_mod.infer_embeddings(
        strings=["gamma"],
        model="gemini-embedding-001",
        gemini_key=None,
        openai_key="legacy-openai-key",
        return_numpy=True,
    )

    assert np.allclose(out, expected)


def test_lt_gemini_missing_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="gemini_key"):
        utils_mod.infer_embeddings(
            strings=["delta"],
            model="gemini-embedding-001",
            gemini_key=None,
            openai_key=None,
        )


def test_lt_predict_rows_with_openai_requires_v1(monkeypatch):
    monkeypatch.setattr(openai, "__version__", "0.28.1")

    with pytest.raises(ValueError, match="Requires openai version 1.0.0 or higher"):
        utils_mod.predict_rows_with_openai(
            strings_col=["sample"],
            model="gpt-3.5-turbo",
            openai_key="dummy",
            openai_params={},
        )


def test_lt_gemini_import_error_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "google.genai", None)  # makes the import fail

    with pytest.raises(ImportError, match=r"google-genai.*linktransformer\[gemini\]"):
        utils_mod.infer_embeddings_with_gemini(
            strings=["hello"],
            model="gemini-embedding-001",
            api_key="dummy",
        )


def _fake_genai(calls):
    """A stand-in for the google.genai module that records embed_content calls."""

    class _Models:
        def embed_content(self, model, contents, config):
            calls.append({"model": model, "contents": list(contents), "task_type": config.task_type})
            return types.SimpleNamespace(
                embeddings=[types.SimpleNamespace(values=[float(len(text)), 1.0]) for text in contents]
            )

    class _Client:
        def __init__(self, api_key):
            assert api_key == "gemini-key"
            self.models = _Models()

    genai = types.ModuleType("google.genai")
    genai.Client = _Client
    genai.types = types.SimpleNamespace(EmbedContentConfig=lambda task_type: types.SimpleNamespace(task_type=task_type))
    return genai


def test_lt_gemini_embeddings_are_batched_in_order(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "google.genai", _fake_genai(calls))
    strings = ["x" * (i % 7) for i in range(250)]

    out = utils_mod.infer_embeddings_with_gemini(strings, model="gemini-embedding-001", api_key="gemini-key")

    assert [len(c["contents"]) for c in calls] == [100, 100, 50]
    assert all(c["model"] == "models/gemini-embedding-001" and c["task_type"] == "RETRIEVAL_DOCUMENT" for c in calls)
    assert out.shape == (250, 2) and out.dtype == np.float32
    assert np.array_equal(out[:, 0], [len(s) for s in strings])
