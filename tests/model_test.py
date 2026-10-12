import json
import os

import numpy as np
import pytest
import transformers
from packaging.version import Version
from sentence_transformers import SentenceTransformer

import linktransformer as lt

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXTS = ["Tech Corporation", "Microsoft Corp", "Alphabet Inc.", "Apple Incorporated"]


@pytest.fixture(scope="module")
def reference_embeddings():
    return SentenceTransformer(MODEL).encode(TEXTS)


def test_lt_linktransformer_matches_sentence_transformer(reference_embeddings):
    # A model-type mismatch makes sentence-transformers rebuild default modules (e.g. dropping Normalize)
    model = lt.LinkTransformer(MODEL)
    assert [type(m).__name__ for m in model] == ["Transformer", "Pooling", "Normalize"]
    assert np.allclose(model.encode(TEXTS), reference_embeddings, atol=1e-6)


def test_lt_saved_model_reloads_identically(tmp_path, reference_embeddings):
    lt.LinkTransformer(MODEL).save(str(tmp_path))

    with open(tmp_path / "config_sentence_transformers.json") as f:
        assert json.load(f).get("model_type", "SentenceTransformer") == "SentenceTransformer"
    assert np.allclose(lt.LinkTransformer(str(tmp_path)).encode(TEXTS), reference_embeddings, atol=1e-6)
    assert np.allclose(SentenceTransformer(str(tmp_path)).encode(TEXTS), reference_embeddings, atol=1e-6)


def test_lt_loads_checkpoints_saved_as_linktransformer_type(tmp_path, reference_embeddings):
    # linktransformer <= 0.1.18 saved checkpoints with model_type "LinkTransformer"
    SentenceTransformer(MODEL).save(str(tmp_path))
    config_path = tmp_path / "config_sentence_transformers.json"
    with open(config_path) as f:
        config = json.load(f)
    config["model_type"] = "LinkTransformer"
    with open(config_path, "w") as f:
        json.dump(config, f)

    assert np.allclose(lt.LinkTransformer(str(tmp_path)).encode(TEXTS), reference_embeddings, atol=1e-6)


def test_lt_build_from_modules():
    from sentence_transformers import models

    word = models.Transformer(MODEL)
    get_dimension = getattr(word, "get_embedding_dimension", None) or word.get_word_embedding_dimension
    model = lt.LinkTransformer(modules=[word, models.Pooling(get_dimension())])
    assert model.encode(["x"]).shape == (1, 384)


def test_lt_load_model_error_is_clear(tmp_path):
    missing = os.path.join(str(tmp_path), "does-not-exist")
    with pytest.raises(ValueError, match="Can't load the model"):
        lt.load_model(missing)


def test_lt_mluke_models_get_a_clear_error_on_transformers_5(tmp_path):
    # transformers 5 mis-tokenizes (5.0-5.1) or cannot load (later) the mLUKE tokenizer used by Japanese models
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "MLukeTokenizer"}))
    with pytest.raises(Exception) as info:
        lt.LinkTransformer(str(tmp_path))
    if Version(transformers.__version__).major >= 5:
        assert isinstance(info.value, RuntimeError) and "transformers>=4.56,<5" in str(info.value)
    else:
        assert "MLukeTokenizer" not in str(info.value)
