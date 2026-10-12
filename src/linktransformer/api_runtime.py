import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 5
    base_sleep_seconds: float = 15.0
    jitter_seconds: float = 0.0


def parse_retry_policy(policy: Optional[Dict[str, Any]], *, default_max_retries: int, default_base_sleep: float) -> RetryPolicy:
    policy = {} if policy is None else dict(policy)
    max_retries = int(policy.get("max_retries", default_max_retries))
    base_sleep_seconds = float(policy.get("base_sleep_seconds", default_base_sleep))
    jitter_seconds = float(policy.get("jitter_seconds", 0.0))
    if max_retries < 1:
        max_retries = 1
    if base_sleep_seconds < 0:
        base_sleep_seconds = 0.0
    if jitter_seconds < 0:
        jitter_seconds = 0.0
    return RetryPolicy(
        max_retries=max_retries,
        base_sleep_seconds=base_sleep_seconds,
        jitter_seconds=jitter_seconds,
    )


def retry_sleep_seconds(attempt: int, policy: RetryPolicy) -> float:
    wait = policy.base_sleep_seconds * (2 ** attempt)
    if policy.jitter_seconds > 0:
        wait += random.uniform(0.0, policy.jitter_seconds)
    return wait


def call_with_retry(fn: Callable[[], T], policy: RetryPolicy) -> T:
    last_error: Optional[Exception] = None
    for attempt in range(policy.max_retries):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if attempt == policy.max_retries - 1:
                raise
            time.sleep(retry_sleep_seconds(attempt, policy))
    if last_error is not None:
        raise last_error
    raise RuntimeError("call_with_retry reached unreachable state")


def append_jsonl(path: Optional[str], record: Dict[str, Any]) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Optional[str]) -> List[Dict[str, Any]]:
    if path is None:
        return []
    target = Path(path)
    if not target.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
    return rows


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_failure(
    path: Optional[str],
    *,
    function_name: str,
    provider: Optional[str],
    model: Optional[str],
    item_key: Any,
    attempt: int,
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "function": function_name,
        "provider": provider,
        "model": model,
        "item_key": item_key,
        "attempt": attempt,
        "error_type": type(error).__name__,
        "error": repr(error),
    }
    if context:
        payload["context"] = context
    append_jsonl(path, payload)
