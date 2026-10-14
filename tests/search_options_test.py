"""search_precision / search_tiling are passed from the public merge functions to the search."""
import os

import pandas as pd
import pytest

import linktransformer as lt
from linktransformer import infer as infer_mod
from linktransformer import search
from linktransformer.data import DATA_DIR_PATH
from linktransformer.modelling.LinkTransformer import LinkTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def model():
    return LinkTransformer(MODEL)


def _toy(name):
    return pd.read_csv(os.path.join(DATA_DIR_PATH, name))


CALLS = {
    "merge": lambda m, **kw: lt.merge(_toy("toy_comp_2.csv"), _toy("toy_comp_1.csv"), on="CompanyName", model=m, **kw),
    "merge_blocking": lambda m, **kw: lt.merge_blocking(_toy("toy_comp_2.csv"), _toy("toy_comp_1.csv"), on="CompanyName",
                                                        model=m, blocking_vars=["Country"], **kw),
    "aggregate_rows": lambda m, **kw: lt.aggregate_rows(_toy("toy_comp_2.csv"), _toy("toy_comp_1.csv"), model=m,
                                                        left_on="CompanyName", right_on="CompanyName", **kw),
    "merge_knn": lambda m, **kw: lt.merge_knn(_toy("toy_comp_2.csv"), _toy("toy_comp_1.csv"), on="CompanyName", model=m, k=3, **kw),
    "merge_range": lambda m, **kw: lt.merge_range(_toy("toy_comp_2.csv"), _toy("toy_comp_1.csv"), on="CompanyName", model=m,
                                                  sim_threshold=0.5, **kw),
}


@pytest.mark.parametrize("name", sorted(CALLS))
def test_lt_search_options_reach_the_search(name, model, monkeypatch):
    seen = []
    for fn in ("knn_search", "range_search"):
        real = getattr(infer_mod, fn)
        monkeypatch.setattr(infer_mod, fn, lambda *a, _real=real, **kw: (seen.append(kw), _real(*a, **kw))[1])
    monkeypatch.setattr(search, "_fp16_available", lambda device: True)  # run the fp16 path on CPU

    expected = CALLS[name](model)
    out = CALLS[name](model, search_precision="fp16", search_tiling=False)

    assert seen[-1] == {"precision": "fp16", "tiled": False}
    assert seen[0] == {"precision": "fp32", "tiled": True}
    # toy_comp_1 has two identical rows (CompanyID 1 and 6); either one is a correct match, and which
    # one comes back may depend on the search settings, so compare everything except the row ids.
    ids = [c for c in out.columns if c.startswith(("CompanyID", "id_lt"))]
    pd.testing.assert_frame_equal(out.drop(columns=ids).reset_index(drop=True),
                                  expected.drop(columns=ids).reset_index(drop=True), atol=1e-5)


@pytest.mark.parametrize("name", ["merge_k_judge", "merge_knn_with_llm", "merge_knn_openai"])
def test_lt_judge_passes_search_options_to_retrieval(name, monkeypatch):
    seen = {}

    def fake_merge_knn(**kwargs):
        seen.update(kwargs)
        return pd.DataFrame({"CompanyName_x": [], "CompanyName_y": [], "score": []})

    monkeypatch.setattr(infer_mod, "merge_knn", fake_merge_knn)
    getattr(lt, name)(pd.DataFrame({"CompanyName": ["a"]}), pd.DataFrame({"CompanyName": ["b"]}), on="CompanyName",
                      knn_sbert_model=MODEL, openai_key="dummy", llm_provider="openai", judge_llm_model="gpt-4o-mini",
                      search_precision="fp16", search_tiling=False)
    assert seen["search_precision"] == "fp16" and seen["search_tiling"] is False


@pytest.mark.parametrize("name", ["merge", "merge_knn", "merge_range"])
def test_lt_bad_search_option_fails_before_embedding(name, monkeypatch):
    def no_embedding(*args, **kwargs):
        raise AssertionError("embedded before checking the search options")

    monkeypatch.setattr(infer_mod, "infer_embeddings", no_embedding)
    with pytest.raises(ValueError, match="precision"):
        CALLS[name](MODEL, search_precision="bf16")
