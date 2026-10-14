"""Progress bars: the progress_bar arguments and the package-wide LINKTRANSFORMER_PROGRESS=0 switch."""
import ast
import json
import pathlib
import types

import numpy as np
import pandas as pd
import pytest

import linktransformer as lt
import linktransformer.infer as infer_mod
import linktransformer.utils as utils_mod
from linktransformer.modelling.LinkTransformer import LinkTransformer
from linktransformer.progress import hf_progress_bars_off_if_disabled, progress_enabled

PACKAGE_DIR = pathlib.Path(lt.__file__).parent


class _RecordingModel(LinkTransformer):
    """A LinkTransformer that loads nothing and records the arguments of encode."""

    def __init__(self):
        self.encode_kwargs = []

    def encode(self, sentences, **kwargs):
        self.encode_kwargs.append(kwargs)
        return np.random.default_rng(0).random((len(sentences), 4), dtype=np.float32)


@pytest.mark.parametrize("value", ["0", "false", "False", "no", "off", " OFF "])
def test_lt_progress_env_var_turns_bars_off(monkeypatch, value):
    monkeypatch.setenv("LINKTRANSFORMER_PROGRESS", value)
    assert progress_enabled() is False and progress_enabled(True) is False


@pytest.mark.parametrize("value", [None, "", "1", "true"])
def test_lt_progress_bars_follow_the_argument_otherwise(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("LINKTRANSFORMER_PROGRESS", raising=False)
    else:
        monkeypatch.setenv("LINKTRANSFORMER_PROGRESS", value)
    assert progress_enabled() is True and progress_enabled(False) is False


def test_lt_infer_embeddings_progress_bar(monkeypatch):
    monkeypatch.delenv("LINKTRANSFORMER_PROGRESS", raising=False)
    model = _RecordingModel()
    lt.infer_embeddings(["a"], model)
    lt.infer_embeddings(["a"], model, progress_bar=False)
    monkeypatch.setenv("LINKTRANSFORMER_PROGRESS", "0")
    lt.infer_embeddings(["a"], model)
    assert [kwargs["show_progress_bar"] for kwargs in model.encode_kwargs] == [True, False, False]


@pytest.mark.parametrize("function, kwargs", [
    ("merge", dict(on="name")),
    ("merge_knn", dict(on="name", k=1)),
    ("merge_range", dict(on="name", sim_threshold=0.0)),
    ("merge_blocking", dict(on="name", blocking_vars=["block"])),
    ("aggregate_rows", dict(left_on="name", right_on="name")),
])
def test_lt_merge_functions_pass_progress_bar_to_the_encoder(monkeypatch, function, kwargs):
    monkeypatch.delenv("LINKTRANSFORMER_PROGRESS", raising=False)
    model = _RecordingModel()
    df = pd.DataFrame({"name": ["Acme", "Globex"], "block": ["a", "a"]})
    getattr(lt, function)(df, df.copy(), model=model, progress_bar=False, **kwargs)
    assert model.encode_kwargs and all(kw["show_progress_bar"] is False for kw in model.encode_kwargs)


@pytest.mark.parametrize("function, kwargs", [
    ("evaluate_pairs", dict(left_on="a", right_on="b")),
    ("all_pair_combos_evaluate", dict(left_on="a", right_on="b")),
    ("cluster_rows", dict(on="a", sim_threshold=0.9)),
    ("dedup_rows", dict(on="a", sim_threshold=0.9)),
])
def test_lt_single_frame_functions_pass_progress_bar_to_the_encoder(monkeypatch, function, kwargs):
    monkeypatch.delenv("LINKTRANSFORMER_PROGRESS", raising=False)
    model = _RecordingModel()
    df = pd.DataFrame({"a": ["Acme", "Globex", "Initech"], "b": ["Acme Inc", "Globex", "Initech"]})
    getattr(lt, function)(df, model=model, progress_bar=False, **kwargs)
    assert model.encode_kwargs and all(kw["show_progress_bar"] is False for kw in model.encode_kwargs)


def test_lt_merge_k_judge_passes_progress_bar_to_retrieval(monkeypatch):
    captured = {}

    def _fake_merge_knn(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"name_x": ["a"], "name_y": ["b"], "score": [0.99]})

    monkeypatch.setattr(infer_mod, "merge_knn", _fake_merge_knn)
    with pytest.warns(UserWarning):
        lt.merge_k_judge(pd.DataFrame({"name": ["a"]}), pd.DataFrame({"name": ["b"]}), on="name", knn_sbert_model="x",
                         openai_key="dummy", judge_llm_model="gpt-4o-mini", judge_skip_upper_thresh=0.9,
                         progress_bar=False)
    assert captured["progress_bar"] is False


def test_lt_env_var_silences_the_package_tqdm_bars(monkeypatch):
    disabled = []
    real_tqdm = utils_mod.tqdm

    def _recording_tqdm(*args, **kwargs):
        disabled.append(kwargs.get("disable"))
        return real_tqdm(*args, **kwargs)

    monkeypatch.setattr(utils_mod, "tqdm", _recording_tqdm)
    monkeypatch.setattr(infer_mod, "tqdm", _recording_tqdm)

    class _Completions:
        def create(self, **kwargs):
            user = kwargs["messages"][1]["content"]
            if '"left"' in user:  # judge
                content = '{"is_match": 1, "confidence": 0.9}'
            elif user.startswith("["):  # batched classification
                content = json.dumps(["Yes"] * len(json.loads(user)))
            else:  # one text per request (predict_rows_with_openai)
                content = "Yes"
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))],
                                         usage=types.SimpleNamespace(total_tokens=1))

    def _client(api_key=None, timeout=None):
        return types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Completions()))

    monkeypatch.setattr(infer_mod.openai, "OpenAI", _client)
    monkeypatch.setattr(utils_mod.openai, "OpenAI", _client)
    monkeypatch.setenv("LINKTRANSFORMER_PROGRESS", "0")

    lt.predict_rows_with_openai(["a", "b"], openai_key="dummy", openai_topic="x")  # its loop used to be unconditional
    lt.classify_rows(pd.DataFrame({"t": ["a", "b"]}), on="t", model="gpt-4o-mini", openai_key="dummy", openai_topic="x")
    lt.judge_merge_k_out(pd.DataFrame({"n_x": ["a"], "n_y": ["b"], "score": [0.5]}), on="n", openai_key="dummy",
                         judge_llm_model="gpt-4o-mini")
    lt.apply_in_chunks(["a", "b"], lambda batch, kwargs: batch)
    assert len(disabled) == 4 and all(flag is True for flag in disabled)


def test_lt_env_var_also_hides_hugging_face_bars_and_restores_them(monkeypatch):
    from transformers.utils import logging as hf_logging
    datasets = pytest.importorskip("datasets")

    initial = (hf_logging.is_progress_bar_enabled(), datasets.are_progress_bars_disabled())
    monkeypatch.delenv("LINKTRANSFORMER_PROGRESS", raising=False)
    with hf_progress_bars_off_if_disabled():
        assert (hf_logging.is_progress_bar_enabled(), datasets.are_progress_bars_disabled()) == initial

    monkeypatch.setenv("LINKTRANSFORMER_PROGRESS", "0")
    with hf_progress_bars_off_if_disabled():
        assert not hf_logging.is_progress_bar_enabled() and datasets.are_progress_bars_disabled()
    assert (hf_logging.is_progress_bar_enabled(), datasets.are_progress_bars_disabled()) == initial

    @hf_progress_bars_off_if_disabled()
    def _load():
        return hf_logging.is_progress_bar_enabled()

    assert _load() is False and hf_logging.is_progress_bar_enabled() == initial[0]


def _progress_call_problems(path: pathlib.Path):
    problems = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            keywords = {kw.arg: kw.value for kw in node.keywords}
            if name in {"tqdm", "trange"} and "disable" not in keywords:
                problems.append(f"{path.name}:{node.lineno} {name}() without disable=")
            value = keywords.get("show_progress_bar")
            if isinstance(value, ast.Constant) and value.value is True:
                problems.append(f"{path.name}:{node.lineno} show_progress_bar=True")
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "show_progress_bar" and \
                        isinstance(value, ast.Constant) and value.value is True:
                    problems.append(f"{path.name}:{node.lineno} 'show_progress_bar': True")
    return problems


def test_lt_every_progress_bar_in_the_package_can_be_disabled():
    problems = [problem for path in sorted(PACKAGE_DIR.rglob("*.py")) for problem in _progress_call_problems(path)]
    assert problems == []
