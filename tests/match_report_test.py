"""lt.match_report: a small JSON-serializable summary of merge output."""
import json
import os

import numpy as np
import pandas as pd
import pytest

import linktransformer as lt
from linktransformer.data import DATA_DIR_PATH


def _assert_plain_python(value):
    """Only dict/list/str/int/float/bool/None, so the report survives json.dumps and needs no numpy to read."""
    if isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, str), key
            _assert_plain_python(item)
    elif isinstance(value, list):
        for item in value:
            _assert_plain_python(item)
    else:
        assert value is None or type(value) in (str, int, float, bool), (value, type(value))


def _knn_like_output():
    """Hand-made merge_knn style output (k=2): left 0 has two good matches, right 2 is the best match of left 1 and 2."""
    return pd.DataFrame({
        "CompanyName_x": ["Tech Corp", "Tech Corp", "InfoTech", "InfoTech", "Info Tech Ltd", "Info Tech Ltd"],
        "id_lt_x": [0, 0, 1, 1, 2, 2],
        "CompanyName_y": ["Tech Corporation", "TechCorp Inc", "InfoTech Soln", "Globex", "InfoTech Soln", "Acme"],
        "id_lt_y": [0, 1, 2, 3, 2, 4],
        "score": [0.95, 0.91, 0.88, 0.30, 0.86, 0.20],
    })


def test_lt_match_report_on_merge_knn_output():
    df1 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_comp_1.csv"))
    df2 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_comp_2.csv"))
    out = lt.merge_knn(df1, df2, on="CompanyName", model="sentence-transformers/all-MiniLM-L6-v2", k=2)

    report = lt.match_report(out, df1, df2, on="CompanyName")

    _assert_plain_python(report)
    json.dumps(report)
    assert report["rows_out"] == len(out) == 2 * len(df1)
    assert report["rows_in"] == {"left": len(df1), "right": len(df2)}
    assert report["columns"]["left_key"] == "id_lt_x" and report["columns"]["right_key"] == "id_lt_y"
    assert report["columns"]["left_text"] == ["CompanyName_x"]
    assert report["score"]["count"] == len(out) and report["score"]["null"] == 0
    assert report["score"]["min"] <= report["score"]["p50"] <= report["score"]["max"]
    assert report["left"]["matched"] == len(df1) and report["left"]["unmatched"] == 0
    assert report["left"]["max_right_per_key"] == 2  # k=2 candidates for every left row
    assert report["threshold"]["source"] == "score_histogram"
    assert 0 < len(report["samples"]) <= 6
    assert report["llm"] is None and report["match_status"] is None


def test_lt_match_report_counts_one_to_many_links_at_threshold():
    report = lt.match_report(_knn_like_output(), df1=4, df2=6, threshold=0.8)

    assert report["threshold"] == {"value": 0.8, "source": "given", "rows_at_or_above": 4, "rows_below": 2}
    # all candidates: every left row has 2 right rows
    assert report["left"]["with_multiple_right"] == 3
    at_threshold = report["left"]["at_threshold"]
    assert at_threshold == {"matched": 3, "unmatched": 1, "with_multiple_right": 1, "max_right_per_key": 2}
    right_at_threshold = report["right"]["at_threshold"]
    assert right_at_threshold == {"matched": 3, "unmatched": 3, "with_multiple_left": 1, "max_left_per_key": 2}
    assert report["right"]["unmatched"] == 1  # right row 5 never appears


def test_lt_match_report_unmatched_rows_of_merge_range():
    # merge_range keeps unmatched left rows with null right columns and a null score
    out = pd.DataFrame({
        "name_x": ["a", "b", "c"], "id_lt_x": [0, 1, 2],
        "name_y": ["a2", None, None], "id_lt_y": [0, np.nan, np.nan],
        "score": [0.9, np.nan, np.nan],
    })
    report = lt.match_report(out, df1=pd.DataFrame({"name": ["a", "b", "c"]}), df2=2)

    assert report["score"]["count"] == 1 and report["score"]["null"] == 2
    assert report["left"]["matched"] == 1 and report["left"]["unmatched"] == 2
    assert report["right"]["matched"] == 1 and report["right"]["unmatched"] == 1


def test_lt_match_report_suggests_threshold_from_llm_judgments():
    out = _knn_like_output()
    out["llm_is_match"] = [1, 1, 1, 0, 1, 0]
    out["llm_confidence"] = [0.9, 0.8, 0.85, 0.95, 0.7, 0.9]
    out["llm_raw_response"] = "{}"

    report = lt.match_report(out)

    assert report["threshold"]["source"] == "llm_judgments"
    assert report["threshold"]["value"] == 0.86  # lowest score the LLM accepted; everything below it was rejected
    llm = report["llm"]
    assert llm["rows_judged"] == 6 and llm["matches"] == 4 and llm["non_matches"] == 2
    assert llm["disagreements_at_threshold"] == {"llm_match_below_threshold": 0, "llm_non_match_at_or_above_threshold": 0}
    assert report["left"]["llm_matches"]["with_multiple_right"] == 1
    assert all("llm_is_match" in sample for sample in report["samples"])


def test_lt_match_report_counts_rows_the_judge_skipped():
    out = _knn_like_output()
    out["llm_is_match"] = [99, 99, 1, 0, np.nan, 0]  # 99: judge_skip_*_thresh; NaN: not judged yet
    out["llm_confidence"] = [99.0, 99.0, 0.9, 0.8, np.nan, 0.7]

    llm = lt.match_report(out, threshold=0.5)["llm"]

    assert llm["rows_skipped_by_score"] == 2 and llm["rows_not_judged"] == 1 and llm["rows_judged"] == 3
    assert llm["confidence"]["p50"] == 0.8  # sentinel 99 confidences are excluded
    assert llm["disagreements_at_threshold"] == {"llm_match_below_threshold": 0, "llm_non_match_at_or_above_threshold": 0}


def test_lt_match_report_histogram_threshold_splits_bimodal_scores():
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0.4, 0.03, 500), rng.normal(0.9, 0.02, 200)])
    out = pd.DataFrame({"id_lt_x": np.arange(700), "id_lt_y": np.arange(700), "score": scores})

    report = lt.match_report(out)

    assert 0.5 < report["threshold"]["value"] < 0.85
    assert report["threshold"]["rows_at_or_above"] == 200


def test_lt_match_report_samples_straddle_the_threshold_and_are_truncated():
    out = _knn_like_output()
    out["CompanyName_y"] = out["CompanyName_y"] + " " + "x" * 200

    report = lt.match_report(out, threshold=0.87, n_samples=4, max_chars=20)

    samples = report["samples"]
    assert [s["score"] for s in samples] == [0.91, 0.88, 0.86, 0.3]  # 2 nearest above, 2 nearest below
    assert [s["side"] for s in samples] == ["at_or_above", "at_or_above", "below", "below"]
    assert all(len(s["right"]) == 20 and s["right"].endswith("…") for s in samples)
    assert samples[0]["left_key"] == 0 and samples[0]["right_key"] == 1


def test_lt_match_report_merge_blocking_output_needs_explicit_keys():
    out = pd.DataFrame({
        "name_x": ["a", "b", "c"], "id_lt_x": [0, 0, np.nan],
        "name_y": ["a2", "b2", None], "id_lt_y": [0, 0, np.nan],
        "score": [0.9, 0.8, np.nan], "match_status": ["matched_block", "matched_block", "left_only_block"],
    })

    report = lt.match_report(out)

    assert report["left"] is None and report["right"] is None
    assert report["match_status"] == {"matched_block": 2, "left_only_block": 1}
    assert any("left_key/right_key" in note for note in report["notes"])

    keyed = lt.match_report(out, left_key="name_x", right_key="name_y")
    assert keyed["left"]["matched"] == 2


def test_lt_match_report_errors_and_empty_input():
    with pytest.raises(ValueError, match="score_col"):
        lt.match_report(pd.DataFrame({"similarity": [0.5]}))
    with pytest.raises(ValueError, match="not found"):
        lt.match_report(_knn_like_output(), left_key="missing", right_key="id_lt_y")

    empty = lt.match_report(_knn_like_output().iloc[:0])
    _assert_plain_python(empty)
    assert empty["rows_out"] == 0 and empty["score"]["p50"] is None and empty["samples"] == []
