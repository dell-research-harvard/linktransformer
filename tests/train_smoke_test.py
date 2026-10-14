"""Fast end-to-end training checks with tiny models; the full training tests are in *train_test.py (marked slow)."""
import json
import os

import numpy as np
import pandas as pd
import pytest

import linktransformer as lt

pytest.importorskip("datasets")  # training needs the `train` extra
pytest.importorskip("accelerate")

TINY_ENCODER = "hf-internal-testing/tiny-random-BertModel"
TINY_CLASSIFIER = "hf-internal-testing/tiny-random-RobertaForSequenceClassification"


@pytest.fixture
def linked_pairs():
    """Matched company names: each row is a (left, right) pair referring to the same firm."""
    names = [f"Firm {chr(65 + i % 26)}{i} Holdings" for i in range(40)]
    return pd.DataFrame({
        "left_name": names,
        "right_name": [n.replace("Holdings", "Hldgs").upper() for n in names],
        "left_id": range(40),
        "right_id": range(40),
    })


def _check_saved_model(path):
    model = lt.LinkTransformer(path)
    assert model.encode(["Firm A0 Holdings"]).shape[0] == 1
    with open(os.path.join(path, "README.md"), encoding="utf8") as f:
        card = f.read()
    assert "linktransformer.github.io" in card and "{" + "MODEL_NAME}" not in card
    assert os.path.isfile(os.path.join(path, "LT_training_config.json"))


def test_lt_train_supcon_with_pooling_layer(linked_pairs, tmp_path):
    path = lt.train_model(
        data=linked_pairs, model_path=TINY_ENCODER,
        left_col_names=["left_name"], right_col_names=["right_name"],
        left_id_name=["left_id"], right_id_name=["right_id"],
        training_args={"num_epochs": 1, "train_batch_size": 8, "add_pooling_layer": True,
                       "model_save_dir": str(tmp_path), "save_val_test_pickles": False},
    )
    _check_saved_model(path)


def test_lt_train_onlinecontrastive(linked_pairs, tmp_path):
    negatives = linked_pairs.assign(right_name=np.roll(linked_pairs["right_name"].to_numpy(), 1),
                                    right_id=np.roll(linked_pairs["right_id"].to_numpy(), 1))
    data = pd.concat([linked_pairs.assign(label=1), negatives.assign(label=0)], ignore_index=True)
    path = lt.train_model(
        data=data, model_path=TINY_ENCODER,
        left_col_names=["left_name"], right_col_names=["right_name"],
        left_id_name=["left_id"], right_id_name=["right_id"], label_col_name="label",
        training_args={"num_epochs": 1, "train_batch_size": 8, "loss_type": "onlinecontrastive",
                       "model_save_dir": str(tmp_path), "save_val_test_pickles": False},
    )
    _check_saved_model(path)


def test_lt_clf_preprocess_shares_label_ids_across_splits_and_uses_tokenizer_sep():
    from transformers import AutoTokenizer
    from linktransformer.train_clf_model import _shared_label_map, get_num_props_labels, preprocess_data

    # text labels first appear in a different order in each split
    train = pd.DataFrame({"head": ["a", "b", "c"], "body": ["x", "y", "z"], "label": ["Riot", "Protest", "Neither"]})
    val = pd.DataFrame({"head": ["d", "e"], "body": ["u", "v"], "label": ["Neither", "Riot"]})
    label_map = _shared_label_map([train, val], "label")
    train_out, num_labels, _, _ = preprocess_data(train, TINY_CLASSIFIER, ["head", "body"], "label", label_map)
    val_out, _, _, _ = preprocess_data(val, TINY_CLASSIFIER, ["head", "body"], "label", label_map)

    assert num_labels == 3
    assert train_out["label"].tolist() == [label_map["Riot"], label_map["Protest"], label_map["Neither"]]
    assert val_out["label"].tolist() == [label_map["Neither"], label_map["Riot"]]
    # several columns are joined with the tokenizer's separator, as classify_rows does at inference
    sep = AutoTokenizer.from_pretrained(TINY_CLASSIFIER).sep_token
    assert train_out["text"].tolist()[0] == f"a{sep}x"

    # class proportions are listed in label-id order, including ids missing from the split
    assert get_num_props_labels(pd.DataFrame({"label": [2, 2, 2, 0]}), num_labels=3) == (3, [0.25, 0.0, 0.75])


def test_lt_train_classifier(tmp_path):
    texts = [f"Protesters marched in city {i}" for i in range(30)] + [f"Quarterly earnings report {i}" for i in range(30)]
    data = pd.DataFrame({"text": texts, "label": [1] * 30 + [0] * 30})

    best_model_path, best_metric, label_map = lt.train_clf_model(
        data=data, model=TINY_CLASSIFIER, on=["text"], label_col_name="label", data_dir=str(tmp_path),
        model_save_dir=str(tmp_path / "clf"), epochs=1, batch_size=8, eval_steps=1, save_steps=1,
    )

    assert best_model_path is not None and os.path.isdir(best_model_path)
    with open(os.path.join(best_model_path, "LT_training_config.json")) as f:
        assert json.load(f)["base_model_path"] == TINY_CLASSIFIER
    out = lt.classify_rows(data.head(4), on="text", model=best_model_path, progress_bar=False)
    assert len(out) == 4 and out["clf_preds_text"].notna().all()
