"""dry_run=True on the LLM functions: count the API calls a run would make, without calling any API."""
import json
import types
import warnings

import pandas as pd
import pytest

import linktransformer as lt
import linktransformer.infer as infer_mod
import linktransformer.utils as utils_mod
from linktransformer.utils import _OPENAI_CHAT_OVERHEAD_TOKENS, _estimate_tokens

LEFT = pd.DataFrame({"CompanyName": ["Tech Corporation", "InfoTech Solutions", "AlphaSoft", "Société Générale"],
                     "Country": ["USA", "USA", "France", "France"]})
RIGHT = pd.DataFrame({"CompanyName": ["Tech Corp", "InfoTech Soln", "AlphaSoft Systems"],
                      "Country": ["USA", "USA", "France"]})


@pytest.fixture
def no_api(monkeypatch):
    """Any attempt to build an OpenAI or Gemini client fails the test."""
    def _fail(*args, **kwargs):
        raise AssertionError("dry_run must not create an API client")

    monkeypatch.setattr(infer_mod.openai, "OpenAI", _fail)
    monkeypatch.setattr(utils_mod.openai, "OpenAI", _fail)
    monkeypatch.setattr(infer_mod, "_gemini_client", _fail)
    monkeypatch.setattr(utils_mod, "_gemini_client", _fail)


@pytest.fixture
def fake_openai(monkeypatch):
    """A fake OpenAI client that records every chat request and answers like the real API would."""
    requests = []

    class _Completions:
        def create(self, **kwargs):
            requests.append(kwargs)
            user = kwargs["messages"][1]["content"]
            if '"left"' in user:  # judge
                content = '{"is_match": 1, "confidence": 0.9}'
            elif "Yes" in kwargs["messages"][0]["content"]:  # classification
                content = json.dumps(["Yes"] * len(json.loads(user)))
            else:  # transform
                content = json.dumps([text.upper() for text in json.loads(user)])
            message = types.SimpleNamespace(content=content)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    def _client(api_key=None, timeout=None):
        return types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Completions()))

    monkeypatch.setattr(infer_mod.openai, "OpenAI", _client)
    monkeypatch.setattr(utils_mod.openai, "OpenAI", _client)
    return requests


def _tokens_sent(requests):
    """The dry-run estimate for requests that were actually sent."""
    return sum(_estimate_tokens(r["messages"][0]["content"]) + _estimate_tokens(r["messages"][1]["content"])
               + _OPENAI_CHAT_OVERHEAD_TOKENS for r in requests)


def _candidates():
    return pd.DataFrame({
        "CompanyName_x": ["A", "B", "C", "D", "E"],
        "CompanyName_y": ["A2", "B2", "C2", "D2", "E2"],
        "score": [0.99, 0.80, 0.70, 0.20, 0.75],
        "llm_is_match": [None, 1, None, None, None],
        "llm_confidence": [None, 0.9, None, None, None],
        "llm_raw_response": [None, "{}", None, None, None],
    })


def test_lt_judge_dry_run_matches_the_real_run(monkeypatch, fake_openai, tmp_path):
    checkpoint = tmp_path / "judge.jsonl"
    checkpoint.write_text(json.dumps({"kind": "merge_k_judge_row", "index": 2, "llm_is_match": 0, "llm_confidence": 0.1,
                                      "llm_raw_response": "{}", "llm_input_payload": ""}) + "\n", encoding="utf-8")
    kwargs = dict(on="CompanyName", openai_key="dummy", judge_llm_model="gpt-4o-mini", checkpoint_path=str(checkpoint),
                  resume=True, judge_skip_lower_thresh=0.3, judge_skip_upper_thresh=0.95)

    report = lt.judge_merge_k_out(_candidates(), dry_run=True, **kwargs)

    assert fake_openai == []
    assert checkpoint.read_text(encoding="utf-8").count("\n") == 1  # dry run writes no checkpoint
    assert report["llm_calls"] == 1 and report["rows_to_send"] == 1 and report["rows_total"] == 5
    assert report["rows_skipped"] == {"already_judged": 1, "resumed_from_checkpoint": 1,
                                      "outside_judge_score_range": 2, "max_retries_zero": 0}
    assert report["max_output_tokens_per_call"] == 50 and report["provider"] == "openai"
    json.dumps(report)

    lt.judge_merge_k_out(_candidates(), progress_bar=False, **kwargs)
    assert len(fake_openai) == report["llm_calls"]
    assert report["estimated_input_tokens"] == _tokens_sent(fake_openai)


def test_lt_judge_dry_run_reports_a_missing_key_instead_of_raising(monkeypatch, no_api):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        lt.judge_merge_k_out(_candidates(), on="CompanyName", judge_llm_model="gpt-4o-mini")

    report = lt.judge_merge_k_out(_candidates(), on="CompanyName", judge_llm_model="gpt-4o-mini", dry_run=True)
    assert report["llm_calls"] == 4
    assert any("No openai API key" in note for note in report["notes"])


def test_lt_merge_k_judge_dry_run_skips_retrieval_without_thresholds(monkeypatch, no_api):
    def _no_retrieval(**kwargs):
        raise AssertionError("merge_knn must not run")

    monkeypatch.setattr(infer_mod, "merge_knn", _no_retrieval)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        report = lt.merge_k_judge(LEFT, RIGHT, on=["CompanyName", "Country"], k=5, openai_key="dummy",
                                  knn_sbert_model="sentence-transformers/all-MiniLM-L6-v2",
                                  judge_llm_model="gpt-4o-mini", dry_run=True)

    assert report["function"] == "merge_k_judge"
    assert report["llm_calls"] == len(LEFT) * len(RIGHT)  # k is capped at len(df2)
    assert report["retrieval"] == {"ran": False, "model": "sentence-transformers/all-MiniLM-L6-v2", "k": 5,
                                   "drop_sim_threshold": None, "candidate_rows": 12}
    assert report["estimated_input_tokens"] > 0


def test_lt_merge_k_judge_dry_run_count_matches_real_run(fake_openai):
    kwargs = dict(on=["CompanyName", "Country"], k=2, openai_key="dummy", progress_bar=False,
                  knn_sbert_model="sentence-transformers/all-MiniLM-L6-v2", judge_llm_model="gpt-4o-mini")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        report = lt.merge_k_judge(LEFT, RIGHT, dry_run=True, **kwargs)
        assert fake_openai == []
        judged = lt.merge_k_judge(LEFT, RIGHT, **kwargs)

    assert report["llm_calls"] == len(fake_openai) == len(judged) == 8
    # same payload shape; only the right rows differ (retrieval was skipped), so the estimate is close, not exact
    assert abs(report["estimated_input_tokens"] - _tokens_sent(fake_openai)) <= 0.25 * _tokens_sent(fake_openai)


def test_lt_merge_k_judge_dry_run_retrieves_when_thresholds_set(monkeypatch, no_api):
    calls = []

    def _fake_merge_knn(**kwargs):
        calls.append(kwargs)
        out = pd.DataFrame({"CompanyName_x": ["A", "A", "B", "B"], "CompanyName_y": ["A2", "C2", "B2", "C2"],
                            "score": [0.97, 0.5, 0.8, 0.6]})
        return out[out["score"] >= kwargs["drop_sim_threshold"]]

    monkeypatch.setattr(infer_mod, "merge_knn", _fake_merge_knn)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        report = lt.merge_k_judge(pd.DataFrame({"CompanyName": ["A", "B"]}), pd.DataFrame({"CompanyName": ["A2", "B2"]}),
                                  on="CompanyName", k=2, openai_key="dummy", knn_sbert_model="x",
                                  judge_llm_model="gpt-4o-mini", drop_sim_threshold=0.55, judge_skip_upper_thresh=0.95,
                                  dry_run=True)

    assert len(calls) == 1 and calls[0]["progress_bar"] is True
    assert report["retrieval"]["ran"] is True and report["retrieval"]["candidate_rows"] == 3
    assert report["llm_calls"] == 2 and report["rows_skipped"]["outside_judge_score_range"] == 1
    assert any("embedding_cache" in note for note in report["notes"])


def test_lt_classify_rows_dry_run_matches_the_real_run(fake_openai, tmp_path):
    df = pd.DataFrame({"text": [f"protest number {i} in the city" for i in range(10)]})
    checkpoint = tmp_path / "clf.jsonl"
    checkpoint.write_text(json.dumps({"kind": "llm_classification_batch", "start": 0, "end": 3, "indices": [0, 1, 2],
                                      "responses": ["Yes", "No", "Yes"]}) + "\n", encoding="utf-8")
    kwargs = dict(on="text", model="gpt-4o-mini", openai_key="dummy", openai_topic="protests", batch_size=3,
                  checkpoint_path=str(checkpoint), resume=True, progress_bar=False)

    report = lt.classify_rows(df, dry_run=True, **kwargs)

    assert fake_openai == []
    assert report["function"] == "classify_rows"
    assert report["llm_calls"] == 3 and report["rows_to_send"] == 7
    assert report["rows_skipped"] == {"resumed_from_checkpoint": 3} and report["rows_per_call"] == 3

    out = lt.classify_rows(df, **kwargs)
    assert len(fake_openai) == report["llm_calls"]
    assert report["estimated_input_tokens"] == _tokens_sent(fake_openai)
    assert out["clf_preds_text"].tolist() == [1, 0, 1] + [1] * 7


def test_lt_classify_rows_dry_run_with_local_model_loads_nothing(monkeypatch):
    def _no_load(*args, **kwargs):
        raise AssertionError("dry_run must not load the classifier")

    monkeypatch.setattr(infer_mod, "load_clf", _no_load)
    report = lt.classify_rows(pd.DataFrame({"text": ["a", "b"]}), on="text", model="some/classifier", dry_run=True)
    assert report["provider"] == "local" and report["llm_calls"] == 0


def test_lt_transform_rows_dry_run_matches_the_real_run(fake_openai):
    df = pd.DataFrame({"name": ["acme", None, "globex", "initech", "umbrella", None, "hooli"]})
    kwargs = dict(on="name", model="gpt-4o-mini", openai_key="dummy", batch_size=2, progress_bar=False)

    report = lt.transform_rows(df, dry_run=True, **kwargs)

    assert fake_openai == []
    assert report["llm_calls"] == 3 and report["rows_to_send"] == 5
    assert report["rows_skipped"] == {"null": 2, "resumed_from_checkpoint": 0}

    out = lt.transform_rows(df, **kwargs)
    assert len(fake_openai) == report["llm_calls"]
    assert report["estimated_input_tokens"] == _tokens_sent(fake_openai)
    assert out["transformed_name"].tolist() == ["ACME", None, "GLOBEX", "INITECH", "UMBRELLA", None, "HOOLI"]


def test_lt_gemini_dry_runs_estimate_the_prompt_sent(monkeypatch):
    prompts = []

    def _fake_generate(client, model, prompt_text, generation_config=None):
        prompts.append(prompt_text)
        inputs = json.loads(prompt_text.rsplit("Inputs:\n", 1)[1])
        return json.dumps(["Yes"] * len(inputs))

    monkeypatch.setattr(utils_mod, "_gemini_client", lambda *args, **kwargs: object())
    monkeypatch.setattr(utils_mod, "_gemini_generate_text", _fake_generate)
    df = pd.DataFrame({"text": ["東京の抗議デモ", "a quiet day", "march downtown", "strike at the port"]})
    kwargs = dict(on="text", model="gemini-2.5-flash", gemini_key="dummy", openai_topic="protests", batch_size=3,
                  progress_bar=False)

    report = lt.classify_rows(df, dry_run=True, **kwargs)
    assert prompts == [] and report["provider"] == "gemini" and report["llm_calls"] == 2

    lt.classify_rows(df, **kwargs)
    sent = sum(_estimate_tokens(prompt) for prompt in prompts)
    assert len(prompts) == 2
    assert abs(report["estimated_input_tokens"] - sent) <= report["llm_calls"]  # rounding of the split prompt
    assert _estimate_tokens("東京") == 2  # non-ASCII characters count one token each
    assert _estimate_tokens(json.dumps(["東京"])) == 9  # payloads carry them as \uXXXX escapes, 4 tokens each


def test_lt_aliases_pass_dry_run_through(monkeypatch, no_api):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for alias in (lt.merge_knn_with_llm, lt.merge_knn_openai):
            report = alias(LEFT, RIGHT, on="CompanyName", k=1, openai_key="dummy", judge_llm_model="gpt-4o-mini",
                           knn_sbert_model="x", dry_run=True)
            assert report["llm_calls"] == len(LEFT)
