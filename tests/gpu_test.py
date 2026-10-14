"""GPU code paths. Skipped without CUDA; the multi-GPU tests need at least two GPUs."""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
import torch

import linktransformer as lt
from linktransformer import infer as infer_mod
from linktransformer import search
from linktransformer.data import DATA_DIR_PATH
from linktransformer.embedding_cache import model_fingerprint
from linktransformer.modelling.LinkTransformer import LinkTransformer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
needs_two_gpus = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TINY_ENCODER = "hf-internal-testing/tiny-random-BertModel"


def _normalized(rng, n, dim=64):
    x = rng.standard_normal((n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_lt_search_on_gpu_matches_cpu(monkeypatch):
    rng = np.random.default_rng(0)
    queries, corpus = _normalized(rng, 300), _normalized(rng, 2000)
    assert search._device().type == "cuda"
    gpu_knn, gpu_range = search.knn_search(queries, corpus, 5), search.range_search(queries, corpus, 0.3)

    monkeypatch.setattr(search, "_device", lambda: torch.device("cpu"))
    cpu_knn, cpu_range = search.knn_search(queries, corpus, 5), search.range_search(queries, corpus, 0.3)

    assert np.array_equal(gpu_knn[1], cpu_knn[1]) and np.allclose(gpu_knn[0], cpu_knn[0], atol=1e-5)
    assert np.array_equal(gpu_range[0], cpu_range[0]) and np.array_equal(gpu_range[2], cpu_range[2])


def _gpu_search_data():
    rng = np.random.default_rng(1)
    corpus = _normalized(rng, 40000)
    corpus[20000:20500] = corpus[:500] + 1e-4 * rng.standard_normal((500, corpus.shape[1])).astype(np.float32)  # near duplicates
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = np.concatenate([corpus[:300], _normalized(rng, 2700)])
    return queries, corpus


@pytest.mark.parametrize("setup", ["default", "small tiles", "untiled", "fp16 output", "forced fallback"])
def test_lt_fp16_search_on_gpu_matches_fp32(setup, monkeypatch):
    if setup == "small tiles":
        monkeypatch.setattr(search, "_CUDA_CORPUS_TILE", 1024)
        monkeypatch.setattr(search, "_CUDA_MAX_SCORES", 1 << 20)
    if setup == "fp16 output":  # torch versions without torch.mm(out_dtype=...)
        monkeypatch.setattr(search, "_fp16_output_is_fp32", lambda device_type: False)
    if setup == "forced fallback":
        monkeypatch.setattr(search, "_FP16_EXTRA_CANDIDATES", 0)
    tiled = setup != "untiled"
    queries, corpus = _gpu_search_data()
    sims = queries.astype(np.float64) @ corpus.astype(np.float64).T

    for k in (1, 10):
        scores, indices = search.knn_search(queries, corpus, k, precision="fp16", tiled=tiled)
        expected = -np.sort(-sims, axis=1)[:, :k]
        assert np.allclose(np.take_along_axis(sims, indices, axis=1), expected, atol=1e-5)
        assert np.allclose(scores, expected, atol=1e-5)
        fp32_scores, fp32_indices = search.knn_search(queries, corpus, k, tiled=tiled)
        assert np.allclose(fp32_scores, expected, atol=1e-5)

    for precision in ("fp16", "fp32"):
        lims, scores, indices = search.range_search(queries, corpus, 0.3, precision=precision, tiled=tiled)
        rows = np.repeat(np.arange(len(queries)), np.diff(lims))
        assert np.allclose(scores, sims[rows, indices], atol=1e-5)
        assert all(np.all(np.diff(indices[lims[i]:lims[i + 1]]) > 0) for i in range(len(queries)))
        # Same matches as float64, except pairs that fp32 rounding can put on either side of the threshold.
        found = np.zeros(sims.shape, dtype=bool)
        found[rows, indices] = True
        differ = found != (sims >= 0.3)
        assert np.all(np.abs(sims[differ] - 0.3) < 1e-5)


def test_lt_fp16_search_rarely_falls_back(caplog):
    import logging

    queries, corpus = _gpu_search_data()
    with caplog.at_level(logging.DEBUG, logger="linktransformer.search"):
        search.knn_search(queries[300:], corpus, 10, precision="fp16")
    assert "searched again in fp32" not in caplog.text


def test_lt_merge_knn_fp16_on_gpu_matches_fp32():
    df1 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_comp_2.csv"))
    df2 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_comp_1.csv"))
    model = LinkTransformer(MODEL)
    fp32 = lt.merge_knn(df1, df2, on="CompanyName", model=model, k=3)
    fp16 = lt.merge_knn(df1, df2, on="CompanyName", model=model, k=3, search_precision="fp16")
    assert fp16["CompanyName_y"].tolist() == fp32["CompanyName_y"].tolist()
    assert np.allclose(fp16["score"], fp32["score"], atol=1e-5)


def test_lt_model_encodes_on_gpu_like_cpu():
    gpu_model, cpu_model = LinkTransformer(MODEL), LinkTransformer(MODEL, device="cpu")
    assert gpu_model.device.type == "cuda"
    texts = ["Tech Corporation", "Microsoft Corp", "Alphabet Inc."]
    assert np.allclose(gpu_model.encode(texts), cpu_model.encode(texts), atol=1e-4)


def test_lt_merge_knn_on_gpu_matches_cpu():
    df1 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_comp_2.csv"))
    df2 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_comp_1.csv"))
    out = {}
    for device in ("cuda", "cpu"):
        out[device] = lt.merge_knn(df1, df2, on="CompanyName", model=LinkTransformer(MODEL, device=device), k=2)
    assert out["cuda"]["CompanyName_y"].tolist() == out["cpu"]["CompanyName_y"].tolist()
    assert np.allclose(out["cuda"]["score"], out["cpu"]["score"], atol=1e-4)


def test_lt_classify_rows_runs_on_gpu(monkeypatch):
    loaded = {}
    original_load_clf = infer_mod.load_clf

    def remembering_load_clf(*args, **kwargs):
        loaded["clf"] = original_load_clf(*args, **kwargs)
        return loaded["clf"]

    monkeypatch.setattr(infer_mod, "load_clf", remembering_load_clf)
    df = pd.DataFrame({"text": ["Protesters gathered outside city hall", "Quarterly earnings rose"]})
    out = lt.classify_rows(df, on="text", model="distilroberta-base", progress_bar=False)
    assert next(loaded["clf"].parameters()).device.type == "cuda"
    assert out["clf_preds_text"].isin([0, 1]).all()


def test_lt_cache_fingerprint_is_device_independent():
    assert model_fingerprint(LinkTransformer(MODEL)) == model_fingerprint(LinkTransformer(MODEL, device="cpu"))


def _linked_pairs(n=40):
    names = [f"Firm {chr(65 + i % 26)}{i} Holdings" for i in range(n)]
    return pd.DataFrame({"left_name": names, "right_name": [s.replace("Holdings", "Hldgs").upper() for s in names],
                         "left_id": range(n), "right_id": range(n)})


def _train_tiny(save_dir):
    return lt.train_model(
        data=_linked_pairs(), model_path=TINY_ENCODER,
        left_col_names=["left_name"], right_col_names=["right_name"],
        left_id_name=["left_id"], right_id_name=["right_id"],
        training_args={"num_epochs": 1, "train_batch_size": 8, "model_save_dir": str(save_dir),
                       "save_val_test_pickles": False},
    )


def test_lt_training_uses_gpu(tmp_path):
    torch.cuda.reset_peak_memory_stats()
    path = _train_tiny(tmp_path)
    assert torch.cuda.max_memory_allocated() > 0
    assert LinkTransformer(path).encode(["Firm A0 Holdings"]).shape[0] == 1


@needs_two_gpus
def test_lt_training_with_several_visible_gpus(tmp_path):
    # Without torchrun, training runs on a single device even though several GPUs are visible
    path = _train_tiny(tmp_path)
    assert os.path.isfile(os.path.join(path, "README.md"))


@needs_two_gpus
def test_lt_ddp_training_with_torchrun(tmp_path):
    script = tmp_path / "ddp_train.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {os.path.dirname(__file__)!r})
        from gpu_test import _train_tiny
        _train_tiny({str(tmp_path / "models")!r})
    """))
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", str(script)],
        capture_output=True, text=True, timeout=1800,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-5000:]
    saved = tmp_path / "models" / "linkage"
    assert (saved / "README.md").is_file() and (saved / "config_sentence_transformers.json").is_file()
    assert LinkTransformer(str(saved)).encode(["Firm A0 Holdings"]).shape[0] == 1
