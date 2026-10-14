"""The on-disk embedding cache (embedding_cache=...), for infer_embeddings and every function that embeds text."""
import os
import types

import numpy as np
import openai
import pandas as pd
import pytest
import torch

import linktransformer as lt
from linktransformer import infer as infer_mod
from linktransformer.data import DATA_DIR_PATH
from linktransformer.modelling.LinkTransformer import LinkTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def model():
    return LinkTransformer(MODEL)


@pytest.fixture
def encoded(monkeypatch):
    """Records every text passed to LinkTransformer.encode."""
    texts = []
    original = LinkTransformer.encode

    def counting_encode(self, sentences, *args, **kwargs):
        texts.extend(sentences)
        return original(self, sentences, *args, **kwargs)

    monkeypatch.setattr(LinkTransformer, "encode", counting_encode)
    return texts


def test_lt_cache_embeds_only_new_texts(model, encoded, tmp_path):
    cache = str(tmp_path / "cache")
    first = lt.infer_embeddings(["Tech Corp", "Acme", "Tech Corp"], model, embedding_cache=cache)
    assert sorted(encoded) == ["Acme", "Tech Corp"]  # duplicates are embedded once

    encoded.clear()
    second = lt.infer_embeddings(["Acme", "Globex", "Tech Corp"], model, embedding_cache=cache)
    assert encoded == ["Globex"]

    uncached = model.encode(["Tech Corp", "Acme", "Globex"])
    assert np.allclose(first, uncached[[0, 1, 0]], atol=1e-6)
    assert np.allclose(second, uncached[[1, 2, 0]], atol=1e-6)
    assert os.path.isfile(os.path.join(cache, "embeddings.sqlite"))


def test_lt_cache_returns_tensor_when_asked(model, tmp_path):
    out = lt.infer_embeddings(["Acme"], model, return_numpy=False, embedding_cache=str(tmp_path))
    assert isinstance(out, torch.Tensor) and out.shape[0] == 1


def test_lt_cache_is_invalidated_when_weights_change(encoded, tmp_path):
    model = LinkTransformer(MODEL)
    lt.infer_embeddings(["Acme"], model, embedding_cache=str(tmp_path))
    with torch.no_grad():  # e.g. after further training
        next(model.parameters()).add_(0.01)
    encoded.clear()
    lt.infer_embeddings(["Acme"], model, embedding_cache=str(tmp_path))
    assert encoded == ["Acme"]


def test_lt_cache_works_for_api_models(monkeypatch, tmp_path):
    requests = []

    class _Embeddings:
        def create(self, input, model):
            requests.append(list(input))
            return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[float(len(t)), 1.0]) for t in input])

    monkeypatch.setattr(openai, "OpenAI", lambda api_key: types.SimpleNamespace(embeddings=_Embeddings()))
    for _ in range(2):
        out = lt.infer_embeddings(["Acme", "Globex Corp"], "text-embedding-3-small", openai_key="dummy",
                                  embedding_cache=str(tmp_path))
    assert requests == [["Acme", "Globex Corp"]]
    assert np.array_equal(out[:, 0], [4.0, 11.0])


def _toy(name):
    return pd.read_csv(os.path.join(DATA_DIR_PATH, name))


FUNCTION_CALLS = {
    "merge": lambda m, c: lt.merge(_toy("toy_comp_2.csv"), _toy("toy_comp_1.csv"), on="CompanyName", model=m, embedding_cache=c),
    "merge_blocking": lambda m, c: lt.merge_blocking(_toy("toy_comp_2.csv"), _toy("toy_comp_1.csv"), on="CompanyName",
                                                     model=m, blocking_vars=["Country"], embedding_cache=c),
    "merge_knn": lambda m, c: lt.merge_knn(_toy("toy_comp_2.csv"), _toy("toy_comp_1.csv"), on="CompanyName", model=m, k=2,
                                           embedding_cache=c),
    "merge_range": lambda m, c: lt.merge_range(_toy("toy_comp_2.csv"), _toy("toy_comp_1.csv"), on="CompanyName", model=m,
                                               sim_threshold=0.5, embedding_cache=c),
    "aggregate_rows": lambda m, c: lt.aggregate_rows(_toy("toy_comp_2.csv"), _toy("toy_comp_1.csv"), model=m,
                                                     left_on="CompanyName", right_on="CompanyName", embedding_cache=c),
    "evaluate_pairs": lambda m, c: lt.evaluate_pairs(_toy("toy_pairs.csv"), model=m, left_on="company_name_1",
                                                     right_on="company_name_2", embedding_cache=c),
    "all_pair_combos_evaluate": lambda m, c: lt.all_pair_combos_evaluate(_toy("toy_pairs.csv"), model=m, left_on="company_name_1",
                                                                         right_on="company_name_2", embedding_cache=c),
    "cluster_rows": lambda m, c: lt.cluster_rows(_toy("company_clusters.csv"), model=m, on="company_name", sim_threshold=0.8,
                                               embedding_cache=c),
    "dedup_rows": lambda m, c: lt.dedup_rows(_toy("company_clusters.csv"), model=m, on="company_name", sim_threshold=0.8,
                                           embedding_cache=c),
}


@pytest.mark.parametrize("name", sorted(FUNCTION_CALLS))
def test_lt_functions_reuse_cached_embeddings(name, model, encoded, tmp_path):
    call = FUNCTION_CALLS[name]
    expected = call(model, None)

    encoded.clear()
    first = call(model, str(tmp_path))
    assert encoded, "the first cached call should embed the texts"
    encoded.clear()
    second = call(model, str(tmp_path))

    assert encoded == [], f"{name} embedded {encoded} again despite the cache"
    for out in (first, second):
        pd.testing.assert_frame_equal(out.reset_index(drop=True), expected.reset_index(drop=True), atol=1e-5)


@pytest.mark.parametrize("name", ["merge_k_judge", "merge_knn_with_llm", "merge_knn_openai"])
def test_lt_judge_passes_cache_to_retrieval(name, monkeypatch):
    seen = {}

    def fake_merge_knn(**kwargs):
        seen.update(kwargs)
        return pd.DataFrame({"CompanyName_x": [], "CompanyName_y": [], "score": []})

    monkeypatch.setattr(infer_mod, "merge_knn", fake_merge_knn)
    getattr(lt, name)(pd.DataFrame({"CompanyName": ["a"]}), pd.DataFrame({"CompanyName": ["b"]}), on="CompanyName",
                      knn_sbert_model=MODEL, openai_key="dummy", llm_provider="openai", judge_llm_model="gpt-4o-mini",
                      embedding_cache="some/folder")
    assert seen["embedding_cache"] == "some/folder"


def test_lt_cache_keeps_finished_chunks_when_interrupted(monkeypatch, model, tmp_path):
    from linktransformer import embedding_cache

    monkeypatch.setattr(embedding_cache, "_EMBED_CHUNK", 2)
    texts = ["a", "b", "c", "d"]
    calls = []

    def flaky_embed(batch):
        calls.append(list(batch))
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return model.encode(batch)

    with pytest.raises(RuntimeError):
        embedding_cache.cached_embeddings(texts, str(tmp_path), "m", flaky_embed)
    calls.clear()
    embedding_cache.cached_embeddings(texts, str(tmp_path), "m", lambda batch: (calls.append(list(batch)), model.encode(batch))[1])
    assert calls == [["c", "d"]]  # the first chunk was kept
