"""Summaries of linkage output, for checking a merge before using it.

match_report needs only numpy and pandas, so it is cheap to import and to run on frames with millions of rows.
"""
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas import DataFrame

_QUANTILES = {"p01": 0.01, "p05": 0.05, "p10": 0.1, "p25": 0.25, "p50": 0.5, "p75": 0.75, "p90": 0.9, "p95": 0.95,
              "p99": 0.99}
# judge_merge_k_out marks rows it did not send to the LLM (judge_skip_*_thresh) with this value
_LLM_SKIPPED_SENTINEL = 99


def _round(value: Any, digits: int = 4) -> Optional[float]:
    value = float(value)
    return None if math.isnan(value) else round(value, digits)


def _jsonable(value: Any) -> Any:
    """A plain Python value (int, float, str, bool or None) for a dataframe cell."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(value) else float(value)
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _as_columns(cols: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
    if cols is None:
        return None
    return [cols] if isinstance(cols, str) else list(cols)


def _row_count(frame: Optional[Union[DataFrame, int]]) -> Optional[int]:
    if frame is None:
        return None
    return int(frame) if isinstance(frame, (int, np.integer)) else len(frame)


def _text_columns(
    df: DataFrame,
    on: Optional[Union[str, List[str]]],
    left_on: Optional[Union[str, List[str]]],
    right_on: Optional[Union[str, List[str]]],
    suffixes: Tuple[str, str],
) -> Tuple[List[str], List[str]]:
    """The output columns holding the left and right text, resolved like merge's on/left_on/right_on."""
    left_cols = _as_columns(left_on) or _as_columns(on)
    right_cols = _as_columns(right_on) or _as_columns(on)
    if left_cols is None and right_cols is None:
        shared = [col[: -len(suffixes[0])] for col in df.columns
                  if col.endswith(suffixes[0]) and f"{col[: -len(suffixes[0])]}{suffixes[1]}" in df.columns]
        left_cols = right_cols = [base for base in shared if base != "id_lt"]
    left_cols, right_cols = left_cols or right_cols or [], right_cols or left_cols or []

    def _resolve(cols: List[str], suffix: str) -> List[str]:
        return [f"{col}{suffix}" if f"{col}{suffix}" in df.columns else col for col in cols
                if f"{col}{suffix}" in df.columns or col in df.columns]

    return _resolve(left_cols, suffixes[0]), _resolve(right_cols, suffixes[1])


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max(max_chars - 1, 0)] + "…"


def _row_text(row: pd.Series, cols: List[str], max_chars: int) -> Optional[str]:
    values = [str(row[col]) for col in cols if _jsonable(row[col]) is not None]
    return _truncate(" | ".join(values), max_chars) if values else None


def _score_summary(scores: np.ndarray, n_null: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"count": int(len(scores)), "null": int(n_null)}
    if len(scores) == 0:
        summary.update({"min": None, **{name: None for name in _QUANTILES}, "max": None, "mean": None})
        return summary
    quantiles = np.quantile(scores, list(_QUANTILES.values()))
    summary["min"] = _round(scores.min())
    summary.update({name: _round(q) for name, q in zip(_QUANTILES, quantiles)})
    summary["max"] = _round(scores.max())
    summary["mean"] = _round(scores.mean())
    return summary


def _threshold_from_llm(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """The score cutoff (match if score >= cutoff) that best agrees with the LLM's labels, by F1."""
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    true_pos = np.cumsum(sorted_labels)
    f1 = 2 * true_pos / (np.arange(1, len(sorted_scores) + 1) + positives)
    # a cutoff can only fall where the score changes
    valid = np.append(sorted_scores[:-1] != sorted_scores[1:], True)
    best = int(np.argmax(np.where(valid, f1, -1.0)))
    return float(sorted_scores[best])


def _threshold_from_histogram(scores: np.ndarray, bins: int = 256) -> Optional[float]:
    """Otsu's split of the score histogram: the cutoff separating a low-score and a high-score group best."""
    low, high = float(scores.min()), float(scores.max())
    if not high > low:
        return None
    counts, edges = np.histogram(scores, bins=bins, range=(low, high))
    centers = (edges[:-1] + edges[1:]) / 2
    weight_low = np.cumsum(counts)[:-1]
    weight_high = len(scores) - weight_low
    sum_low = np.cumsum(counts * centers)[:-1]
    mean_low = sum_low / np.maximum(weight_low, 1)
    mean_high = (np.sum(counts * centers) - sum_low) / np.maximum(weight_high, 1)
    between = weight_low * weight_high * (mean_low - mean_high) ** 2
    # every cut inside an empty gap between the groups scores the same: take the middle of the gap
    first = last = int(np.argmax(between))
    while last + 1 < len(between) and between[last + 1] == between[first]:
        last += 1
    return float((edges[first + 1] + edges[last + 1]) / 2)


def _key_links(pairs: DataFrame, key_cols: List[str]) -> Dict[str, int]:
    """For the (left key, right key) pairs in `pairs`: distinct keys of one side, and how many link to several keys of the other."""
    if pairs.empty:
        return {"matched": 0, "with_multiple": 0, "max_per_key": 0}
    counts = pairs.drop_duplicates().groupby(key_cols, sort=False, dropna=False).size()
    return {"matched": int(len(counts)), "with_multiple": int((counts > 1).sum()), "max_per_key": int(counts.max())}


def _side_summary(
    pairs: DataFrame,
    subsets: Dict[str, Optional[DataFrame]],
    key_cols: List[str],
    rows_in: Optional[int],
    other: str,
) -> Dict[str, Any]:
    """Match counts for one side over all scored pairs, plus the same counts within each named subset of pairs."""
    def _section(subset: DataFrame) -> Dict[str, Any]:
        links = _key_links(subset, key_cols)
        return {
            "matched": links["matched"],
            "unmatched": None if rows_in is None else max(rows_in - links["matched"], 0),
            f"with_multiple_{other}": links["with_multiple"],
            f"max_{other}_per_key": links["max_per_key"],
        }

    summary = _section(pairs)
    summary.update({name: None if subset is None else _section(subset) for name, subset in subsets.items()})
    return summary


def _nearest_rows(distance: np.ndarray, candidates: np.ndarray, n: int) -> np.ndarray:
    """The n entries of `candidates` (row positions) with the smallest distance."""
    if n <= 0 or len(candidates) == 0:
        return candidates[:0]
    if n < len(candidates):
        candidates = candidates[np.argpartition(distance[candidates], n - 1)[:n]]
    return candidates


def match_report(
    df: DataFrame,
    df1: Optional[Union[DataFrame, int]] = None,
    df2: Optional[Union[DataFrame, int]] = None,
    on: Optional[Union[str, List[str]]] = None,
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    score_col: str = "score",
    threshold: Optional[float] = None,
    suffixes: Tuple[str, str] = ("_x", "_y"),
    left_key: Optional[Union[str, List[str]]] = None,
    right_key: Optional[Union[str, List[str]]] = None,
    n_samples: int = 6,
    max_chars: int = 80,
) -> Dict[str, Any]:
    """
    Summarize the output of merge, merge_knn, merge_range, merge_blocking or merge_k_judge in a small dict of plain
    Python values (JSON-serializable), to check a linkage and pick a score threshold without printing the dataframe.
    Runs in about a second on a million rows.

    :param df (DataFrame): The merge output.
    :param df1 (DataFrame or int, optional): The left input, or its row count. Needed to count unmatched left rows.
    :param df2 (DataFrame or int, optional): The right input, or its row count. Needed to count unmatched right rows.
    :param on (Union[str, List[str]], optional): Column(s) shown as the text of each sample pair, as passed to the merge.
        Defaults to the columns present with both suffixes.
    :param left_on (Union[str, List[str]], optional): Left text column(s) for samples. Defaults to on.
    :param right_on (Union[str, List[str]], optional): Right text column(s) for samples. Defaults to on.
    :param score_col (str): Similarity column. Defaults to "score".
    :param threshold (float, optional): Score at or above which a pair counts as a match. If None, one is suggested:
        the cutoff that agrees best (F1) with the LLM judgments when `llm_is_match` is present, otherwise the
        Otsu split of the score histogram. A suggestion is a starting point for review, not a calibrated cutoff.
    :param suffixes (Tuple[str, str]): Suffixes used by the merge. Defaults to ('_x', '_y').
    :param left_key (Union[str, List[str]], optional): Column(s) identifying a left row. Defaults to "id_lt_x", which
        merge, merge_knn, merge_range and merge_k_judge add. merge_blocking numbers id_lt within each block, so pass
        your own key columns for its output.
    :param right_key (Union[str, List[str]], optional): Column(s) identifying a right row. Defaults to "id_lt_y".
    :param n_samples (int): Number of sample pairs nearest the threshold, half above and half below. Defaults to 6.
    :param max_chars (int): Truncate sample texts to this many characters. Defaults to 80.
    :return: Dict with keys rows_out, rows_in, columns, score (quantiles), threshold, left and right (per-key match
        counts, unmatched counts when df1/df2 are given, and keys linked to several keys of the other side: over all
        scored pairs, at the threshold, and among pairs the LLM judged a match), match_status (merge_blocking),
        llm (judge statistics), samples and notes.
    """
    if score_col not in df.columns:
        raise ValueError(f"Score column {score_col!r} not found in df; pass score_col= to name it.")

    notes: List[str] = []
    rows_in_left, rows_in_right = _row_count(df1), _row_count(df2)

    scores_all = pd.to_numeric(df[score_col], errors="coerce").to_numpy(dtype=float)
    scored_mask = ~np.isnan(scores_all)
    scored_positions = np.flatnonzero(scored_mask)
    scores = scores_all[scored_mask]

    # LLM judge columns from merge_k_judge / judge_merge_k_out
    llm_summary = None
    llm_labels = None
    if "llm_is_match" in df.columns:
        llm_labels = pd.to_numeric(df["llm_is_match"], errors="coerce").to_numpy(dtype=float)

    # threshold: given, or suggested
    threshold_source = "given" if threshold is not None else None
    if threshold is None and len(scores) > 0:
        if llm_labels is not None:
            judged = scored_mask & np.isin(llm_labels, (0, 1))
            threshold = _threshold_from_llm(scores_all[judged], llm_labels[judged])
            if threshold is not None:
                threshold_source = "llm_judgments"
        if threshold is None:
            threshold = _threshold_from_histogram(scores)
            if threshold is not None:
                threshold_source = "score_histogram"
        if threshold is not None:
            notes.append(f"No threshold given; suggested {threshold:.4f} from {threshold_source}. "
                         "Check the samples before relying on it.")
    above_mask = scored_mask & (scores_all >= threshold) if threshold is not None else None

    # keys identifying left and right rows
    auto_keys = left_key is None and right_key is None
    left_cols = _as_columns(left_key) or [f"id_lt{suffixes[0]}"]
    right_cols = _as_columns(right_key) or [f"id_lt{suffixes[1]}"]
    keys_found = all(col in df.columns for col in left_cols + right_cols)
    if not keys_found:
        if not auto_keys:
            missing = [col for col in left_cols + right_cols if col not in df.columns]
            raise ValueError(f"Key column(s) {missing} not found in df.")
        notes.append(f"No {left_cols[0]}/{right_cols[0]} columns: pass left_key/right_key for per-row match counts.")
    elif auto_keys and "match_status" in df.columns:
        keys_found = False
        notes.append("match_status column found (merge_blocking output), whose id_lt numbers restart in every block: "
                     "pass left_key/right_key for per-row match counts.")

    left_summary = right_summary = None
    if keys_found:
        key_frame = df[left_cols + right_cols]
        pairs = key_frame[scored_mask]
        subsets = {
            "at_threshold": key_frame[above_mask] if above_mask is not None else None,
            "llm_matches": key_frame[llm_labels == 1] if llm_labels is not None else None,
        }
        left_summary = _side_summary(pairs, subsets, left_cols, rows_in_left, "right")
        right_summary = _side_summary(pairs, subsets, right_cols, rows_in_right, "left")
        if rows_in_left is None or rows_in_right is None:
            notes.append("Pass df1 and df2 (or their row counts) to count unmatched rows.")

    match_status = None
    if "match_status" in df.columns:
        match_status = {str(status): int(count) for status, count in df["match_status"].value_counts(dropna=False).items()}

    if llm_labels is not None:
        judged = np.isin(llm_labels, (0, 1))
        n_judged = int(judged.sum())
        n_matches = int((llm_labels == 1).sum())
        llm_summary = {
            "rows_judged": n_judged,
            "rows_skipped_by_score": int((llm_labels == _LLM_SKIPPED_SENTINEL).sum()),
            "rows_not_judged": int(np.isnan(llm_labels).sum()),
            "matches": n_matches,
            "non_matches": n_judged - n_matches,
            "match_rate": _round(n_matches / n_judged) if n_judged else None,
            "confidence": None,
            "disagreements_at_threshold": None,
        }
        if "llm_confidence" in df.columns and n_judged:
            confidence = pd.to_numeric(df["llm_confidence"], errors="coerce").to_numpy(dtype=float)[judged]
            confidence = confidence[~np.isnan(confidence)]
            if len(confidence):
                llm_summary["confidence"] = {
                    "p10": _round(np.quantile(confidence, 0.1)), "p50": _round(np.quantile(confidence, 0.5)),
                    "p90": _round(np.quantile(confidence, 0.9)), "mean": _round(confidence.mean()),
                }
        if above_mask is not None:
            llm_summary["disagreements_at_threshold"] = {
                "llm_match_below_threshold": int((scored_mask & ~above_mask & (llm_labels == 1)).sum()),
                "llm_non_match_at_or_above_threshold": int((above_mask & (llm_labels == 0)).sum()),
            }

    # sample pairs nearest the threshold, half on each side
    samples: List[Dict[str, Any]] = []
    left_text, right_text = _text_columns(df, on, left_on, right_on, suffixes)
    if threshold is not None and n_samples > 0 and len(scored_positions) > 0:
        distance = np.abs(scores_all - threshold)
        above_positions = scored_positions[scores_all[scored_positions] >= threshold]
        below_positions = scored_positions[scores_all[scored_positions] < threshold]
        n_below = min(n_samples // 2, len(below_positions))
        n_above = min(n_samples - n_below, len(above_positions))
        n_below = min(n_samples - n_above, len(below_positions))
        chosen = np.concatenate([_nearest_rows(distance, above_positions, n_above),
                                 _nearest_rows(distance, below_positions, n_below)])
        chosen = chosen[np.argsort(-scores_all[chosen], kind="stable")]
        for position in chosen:
            row = df.iloc[int(position)]
            sample: Dict[str, Any] = {
                "score": _round(scores_all[position]),
                "side": "at_or_above" if scores_all[position] >= threshold else "below",
                "left": _row_text(row, left_text, max_chars),
                "right": _row_text(row, right_text, max_chars),
            }
            if keys_found:
                sample["left_key"] = [_jsonable(row[col]) for col in left_cols] if len(left_cols) > 1 else _jsonable(row[left_cols[0]])
                sample["right_key"] = [_jsonable(row[col]) for col in right_cols] if len(right_cols) > 1 else _jsonable(row[right_cols[0]])
            for col in ("llm_is_match", "llm_confidence"):
                if col in df.columns:
                    sample[col] = _jsonable(row[col])
            samples.append(sample)
        if not left_text and not right_text:
            notes.append("No text columns found for the samples: pass on= or left_on=/right_on=.")

    rows_at_or_above = int(above_mask.sum()) if above_mask is not None else None
    return {
        "rows_out": int(len(df)),
        "rows_in": {"left": rows_in_left, "right": rows_in_right},
        "columns": {
            "score": score_col,
            "left_key": (left_cols if len(left_cols) > 1 else left_cols[0]) if keys_found else None,
            "right_key": (right_cols if len(right_cols) > 1 else right_cols[0]) if keys_found else None,
            "left_text": left_text,
            "right_text": right_text,
        },
        "score": _score_summary(scores, int((~scored_mask).sum())),
        "threshold": {
            "value": _round(threshold) if threshold is not None else None,
            "source": threshold_source,
            "rows_at_or_above": rows_at_or_above,
            "rows_below": int(len(scores)) - rows_at_or_above if rows_at_or_above is not None else None,
        },
        "left": left_summary,
        "right": right_summary,
        "match_status": match_status,
        "llm": llm_summary,
        "samples": samples,
        "notes": notes,
    }
