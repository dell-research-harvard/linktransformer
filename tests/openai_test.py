"""OpenAI helpers, tested against fake clients (no API key needed)."""
import types

import httpx
import numpy as np
import openai
import pytest

import linktransformer.utils as utils_mod


def _bad_request(param):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.BadRequestError(
        f"Unsupported parameter: '{param}'",
        response=httpx.Response(400, request=request),
        body={"param": param, "code": "unsupported_parameter"},
    )


class _FakeChat:
    """Rejects the parameters in `unsupported`, like a reasoning model would, and records each call."""

    def __init__(self, unsupported=(), answer="Yes"):
        self.unsupported, self.answer, self.calls = set(unsupported), answer, []
        self.completions = self

    def create(self, model, messages, **kwargs):
        self.calls.append(kwargs)
        for param in kwargs:
            if param in self.unsupported:
                raise _bad_request(param)
        message = types.SimpleNamespace(content=self.answer)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)],
                                     usage=types.SimpleNamespace(total_tokens=3))


def test_lt_chat_completion_maps_max_tokens_and_drops_unsupported_params():
    client = types.SimpleNamespace(chat=_FakeChat(unsupported={"temperature", "top_p"}))

    utils_mod._chat_completion(client, "reasoning-model", [], params={"max_tokens": 20, "top_p": 0.5},
                               defaults={"temperature": 0})

    assert client.chat.calls[0] == {"temperature": 0, "top_p": 0.5, "max_completion_tokens": 20}
    assert client.chat.calls[-1] == {"max_completion_tokens": 20}


def test_lt_chat_completion_raises_other_bad_requests():
    client = types.SimpleNamespace(chat=_FakeChat(unsupported={"messages"}))
    client.chat.create = lambda **kwargs: (_ for _ in ()).throw(_bad_request("messages"))
    with pytest.raises(openai.BadRequestError):
        utils_mod._chat_completion(client, "model", [], defaults={"temperature": 0})


def test_lt_retryable_errors():
    assert utils_mod._is_retryable(RuntimeError("connection reset"))
    assert not utils_mod._is_retryable(_bad_request("temperature"))
    rate_limited = openai.RateLimitError(
        "slow down", response=httpx.Response(429, request=httpx.Request("POST", "https://x")), body=None)
    assert utils_mod._is_retryable(rate_limited)


@pytest.mark.parametrize("answer, expected", [("Yes", 1), ("yes.", 1), (" No ", 0), ("Maybe", None)])
def test_lt_match_label(answer, expected):
    assert utils_mod._match_label(answer, {"Yes": 1, "No": 0}) == expected


def test_lt_predict_rows_does_not_retry_bad_requests(monkeypatch):
    calls = []

    def failing_completion(*args, **kwargs):
        calls.append(1)
        raise _bad_request("model")

    monkeypatch.setattr(utils_mod, "get_completion_from_messages", failing_completion)
    with pytest.raises(openai.BadRequestError):
        utils_mod.predict_rows_with_openai(["text"], openai_key="dummy", openai_topic="protests",
                                           openai_params={}, ratelimit_sleep_time=0)
    assert len(calls) == 1


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

    monkeypatch.setattr(openai, "OpenAI", lambda api_key: types.SimpleNamespace(embeddings=_Embeddings()))
    strings = ["x" * 1200 for _ in range(9)]  # 10,800 characters -> batches of at most 5000

    out = utils_mod.infer_embeddings(strings, "text-embedding-3-small", openai_key="dummy")

    assert [len(r) for r in requests] == [4, 4, 1]
    assert all(sum(len(t) for t in r) <= 5000 for r in requests)
    assert out.shape == (9, 2) and np.all(out[:, 0] == 1200)
