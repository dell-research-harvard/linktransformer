"""OpenAI helpers and retry behaviour, tested against fake clients (no API key needed)."""
import types

import httpx
import numpy as np
import openai
import pytest

import linktransformer.utils as utils_mod
from linktransformer import api_runtime


def _bad_request(param, code="unsupported_parameter"):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.BadRequestError(
        f"Error code: 400 - this model does not support {param}",
        response=httpx.Response(400, request=request),
        body={"param": param, "code": code},
    )


class _FakeChat:
    """Rejects the parameters in `unsupported`, like a reasoning model would, and records each call."""

    def __init__(self, unsupported=()):
        self.unsupported, self.calls = set(unsupported), []
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        for param in kwargs:
            if param in self.unsupported:
                raise _bad_request(param)
        message = types.SimpleNamespace(content="Yes")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def test_lt_resilient_chat_drops_params_named_by_the_api():
    # the error text does not match any of the regexes, so this relies on the structured `param` field
    client = types.SimpleNamespace(chat=_FakeChat(unsupported={"temperature", "top_p"}))
    with pytest.warns(UserWarning, match="adjusted unsupported params"):
        utils_mod._openai_chat_completion_resilient(
            client, {"model": "reasoning-model", "messages": [], "temperature": 0, "top_p": 0.5})
    assert set(client.chat.calls[-1]) == {"model", "messages"}


def test_lt_resilient_chat_raises_other_bad_requests():
    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(
        create=lambda **kwargs: (_ for _ in ()).throw(_bad_request("messages", code="invalid_value")))))
    with pytest.raises(openai.BadRequestError):
        utils_mod._openai_chat_completion_resilient(client, {"model": "m", "messages": []})


def test_lt_retryable_errors():
    assert api_runtime.is_retryable(RuntimeError("connection reset"))
    assert not api_runtime.is_retryable(_bad_request("temperature"))
    rate_limited = openai.RateLimitError(
        "slow down", response=httpx.Response(429, request=httpx.Request("POST", "https://x")), body=None)
    assert api_runtime.is_retryable(rate_limited)


def test_lt_call_with_retry_does_not_retry_bad_requests(monkeypatch):
    calls = []
    monkeypatch.setattr(api_runtime.time, "sleep", lambda seconds: None)

    def failing():
        calls.append(1)
        raise _bad_request("model", code="model_not_found")

    with pytest.raises(openai.BadRequestError):
        api_runtime.call_with_retry(failing, api_runtime.RetryPolicy(max_retries=5, base_sleep_seconds=1))
    assert len(calls) == 1


@pytest.mark.parametrize("answer, expected", [("Yes", "Yes"), ("yes.", "Yes"), (' "No" ', "No"), ("Maybe", "Maybe")])
def test_lt_canonicalize_label(answer, expected):
    assert utils_mod._canonicalize_label(answer, {"Yes": 1, "No": 0}) == expected


def test_lt_predict_rows_maps_answers_to_labels(monkeypatch):
    answers = iter(["Yes.", "no"])
    monkeypatch.setattr(utils_mod, "get_completion_from_messages", lambda *a, **k: (next(answers), 1))
    preds = utils_mod.predict_rows_with_openai(["a", "b"], openai_key="dummy", openai_topic="protests", openai_params={})
    assert preds == [1, 0]


def test_lt_openai_embeddings_are_batched_in_order(monkeypatch):
    requests = []

    class _Embeddings:
        def create(self, input, model):
            requests.append(list(input))
            return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[float(len(t)), 1.0]) for t in input])

    monkeypatch.setattr(utils_mod.openai, "OpenAI", lambda api_key: types.SimpleNamespace(embeddings=_Embeddings()))
    strings = ["x" * 1200 for _ in range(9)]  # 10,800 characters -> batches of at most 5000

    out = utils_mod.infer_embeddings(strings, "text-embedding-3-small", openai_key="dummy")

    assert [len(r) for r in requests] == [4, 4, 1]
    assert all(sum(len(t) for t in r) <= 5000 for r in requests)
    assert out.shape == (9, 2) and np.all(out[:, 0] == 1200)
