import json
import re
import warnings

import pandas as pd
import pytest
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import linktransformer as lt
from linktransformer.model_card_templates import render_classifier_card, render_linktransformer_card

PLACEHOLDER = re.compile(r"\{[A-Z][A-Z_]+\}")


def _front_matter(card):
    return card.split("---")[1]


def test_lt_linktransformer_card_is_complete():
    card = render_linktransformer_card(
        model_name="org/lt-model", num_dimensions=384, architecture="LinkTransformer(...)",
        base_model="sentence-transformers/all-MiniLM-L6-v2", languages=["en", "es"],
        training_config={"loss_type": "onlinecontrastive", "num_epochs": 2},
    )
    assert not PLACEHOLDER.search(card)
    front = _front_matter(card)
    assert "library_name: sentence-transformers" in front
    assert "base_model: sentence-transformers/all-MiniLM-L6-v2" in front
    assert "language:\n- en\n- es" in front
    assert 'lt.merge(df1, df2, on="CompanyName", model="org/lt-model")' in card
    assert "online contrastive loss, epochs 2" in card
    assert "aclanthology.org/2024.acl-demos.21" in card
    assert "```\nLinkTransformer(...)\n```" in card  # architecture block has both fences


def test_lt_card_without_optional_fields():
    card = render_linktransformer_card(model_name="./local-model", num_dimensions=384, architecture="x")
    assert not PLACEHOLDER.search(card)
    assert "base_model:" not in card and "language:" not in card


def test_lt_classifier_card_is_complete():
    card = render_classifier_card(model_name="org/clf", base_model="distilroberta-base", label_map={"no": 0, "yes": 1})
    assert not PLACEHOLDER.search(card)
    assert "library_name: transformers" in _front_matter(card)
    assert "base_model: distilroberta-base" in _front_matter(card)
    assert "- yes: 1" in card
    assert "lt.train_clf_model(" in card
    assert card.count("## Training") == 1


def test_lt_trained_model_gets_linktransformer_card(tmp_path):
    model = lt.LinkTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model.save(str(tmp_path / "untrained"))
    # Not trained: like sentence-transformers, the card loaded from the Hub is kept
    assert "linktransformer.github.io" not in (tmp_path / "untrained" / "README.md").read_text()

    model.model_card_data.add_tags("generated_from_trainer")  # what the trainer does when training starts
    save_dir = tmp_path / "trained"
    save_dir.mkdir()
    with open(save_dir / "LT_training_config.json", "w") as f:
        json.dump({"base_model_path": "sentence-transformers/all-MiniLM-L6-v2", "loss_type": "supcon"}, f)
    model.save(str(save_dir), model_name="org/my-linker")

    card = (save_dir / "README.md").read_text()
    assert "linktransformer.github.io" in card and not PLACEHOLDER.search(card)
    assert "base_model: sentence-transformers/all-MiniLM-L6-v2" in card
    assert "supervised contrastive (SupCon) loss" in card


def test_lt_classifier_card_reports_base_model(tmp_path):
    ckpt = tmp_path / "ckpt"
    AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2).save_pretrained(ckpt)
    AutoTokenizer.from_pretrained("distilroberta-base").save_pretrained(ckpt)
    with open(ckpt / "LT_training_config.json", "w") as f:
        json.dump({"base_model_path": "distilroberta-base"}, f)

    lt.LinkTransformerClassifier(str(ckpt), label_map={"no": 0, "yes": 1}).save(str(tmp_path / "out"), model_name="org/clf")

    card = (tmp_path / "out" / "README.md").read_text()
    assert "base_model: distilroberta-base" in card and not PLACEHOLDER.search(card)


def test_lt_aggregate_rows_does_not_warn_about_merge_type():
    df = pd.DataFrame({"fine": ["apple juice", "steel bolts"]})
    ref = pd.DataFrame({"coarse": ["beverages", "hardware"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        out = lt.aggregate_rows(df, ref, model="sentence-transformers/all-MiniLM-L6-v2", left_on="fine", right_on="coarse")
    assert list(out["coarse"]) == ["beverages", "hardware"]
