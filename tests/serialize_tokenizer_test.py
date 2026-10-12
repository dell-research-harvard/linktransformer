import json
import warnings

import pandas as pd
import pytest
import transformers
from transformers import AutoModelForSequenceClassification

import linktransformer as lt
from linktransformer import utils as utils_mod

DF = pd.DataFrame({"name": ["Tech Corp", None], "country": ["USA", "Canada"]})


def test_lt_explicit_sep_token_wins():
    assert lt.serialize_columns(DF, ["name", "country"], sep_token=" ") == ["Tech Corp USA", " Canada"]
    assert lt.serialize_columns(DF, ["name", "country"], sep_token=" | ", model="sentence-transformers/all-MiniLM-L6-v2") \
        == ["Tech Corp | USA", " | Canada"]


@pytest.mark.parametrize("model", ["text-embedding-3-small", "gemini-embedding-001"])
def test_lt_api_models_do_not_query_hugging_face(monkeypatch, model):
    def no_hub(*args, **kwargs):
        raise AssertionError("serialize_columns should not look up a tokenizer for API models")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", no_hub)
    assert lt.serialize_columns(DF, ["name", "country"], model=model) == ["Tech Corp<SEP>USA", "<SEP>Canada"]


def test_lt_hugging_face_model_sep_token():
    assert lt.serialize_columns(DF, ["name", "country"], model="sentence-transformers/all-MiniLM-L6-v2")[0] == "Tech Corp[SEP]USA"


@pytest.fixture(scope="module")
def checkpoint_without_tokenizer(tmp_path_factory):
    """A classifier saved the way a plain transformers Trainer saves it: weights and config, no tokenizer."""
    path = tmp_path_factory.mktemp("clf")
    AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2).save_pretrained(path)
    return path


def test_lt_tokenizer_from_recorded_base_model(checkpoint_without_tokenizer, tmp_path):
    with open(checkpoint_without_tokenizer / "LT_training_config.json", "w") as f:
        json.dump({"base_model_path": "distilroberta-base"}, f)
    try:
        with pytest.warns(UserWarning, match="base model"):
            tokenizer = utils_mod._load_tokenizer(str(checkpoint_without_tokenizer))
        assert len(tokenizer) > 50000  # the real vocabulary, not an empty placeholder tokenizer

        df = pd.DataFrame({"text": ["Protesters gathered outside city hall", "Quarterly earnings rose"]})
        out = lt.classify_rows(df, on="text", model=str(checkpoint_without_tokenizer), progress_bar=False)
        assert out["clf_preds_text"].isin([0, 1]).all()
    finally:
        (checkpoint_without_tokenizer / "LT_training_config.json").unlink()


def test_lt_explicit_tokenizer(checkpoint_without_tokenizer):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no fallback warning when the tokenizer is given
        tokenizer = utils_mod._load_tokenizer(str(checkpoint_without_tokenizer), tokenizer="distilroberta-base")
    assert len(tokenizer) > 50000


def test_lt_missing_tokenizer_raises_clear_error(checkpoint_without_tokenizer):
    with pytest.raises(OSError, match="tokenizer="):
        utils_mod._load_tokenizer(str(checkpoint_without_tokenizer))
