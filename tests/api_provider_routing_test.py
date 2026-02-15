import builtins
import importlib.machinery
import sys
import types

import numpy as np
import pytest


try:
    import faiss  # noqa: F401
except ModuleNotFoundError:
    if "faiss" not in sys.modules:
        faiss_stub = types.ModuleType("faiss")
        faiss_stub.IndexFlatIP = object
        faiss_stub.__spec__ = importlib.machinery.ModuleSpec(name="faiss", loader=None)
        sys.modules["faiss"] = faiss_stub


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
