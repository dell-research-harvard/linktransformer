import os
import builtins
import sys
import types
import pandas as pd
import numpy as np
from linktransformer import DATA_DIR_PATH
import linktransformer as lt
import linktransformer.infer as infer_mod

import pytest


def test_lt_data_dir_path():
    print(DATA_DIR_PATH)
    assert os.path.exists(DATA_DIR_PATH)
    assert os.path.isdir(DATA_DIR_PATH)

def test_lt_merge():
    df1 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_comp_1.csv"))
    df2 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_comp_2.csv"))

    # Test your function here
    df_lm_matched = lt.merge(df2, df1, merge_type='1:m', on="CompanyName", model="sentence-transformers/all-MiniLM-L6-v2", left_on=None, right_on=None)
    assert isinstance(df_lm_matched, pd.DataFrame)
    # Add more assertions to check the correctness of the output

def test_lt_merge_suffixes():
    df1 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_comp_1.csv"))
    df2 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_comp_2.csv"))

    # Test your function here
    df_lm_matched = lt.merge(df2, df1, merge_type='1:m', on="CompanyName", model="sentence-transformers/all-MiniLM-L6-v2", left_on=None, right_on=None,suffixes=("_1","_2"))
    
    ##Check if the suffixes are added
    assert "CompanyName_1" in df_lm_matched.columns
    assert "CompanyName_2" in df_lm_matched.columns
    assert isinstance(df_lm_matched, pd.DataFrame)
    # Add more assertions to check the correctness of the output




def test_lt_aggregate_rows():
    df_coarse = pd.read_csv(os.path.join(DATA_DIR_PATH, "coarse.csv"))
    df_fine = pd.read_csv(os.path.join(DATA_DIR_PATH, "fine.csv"))

    # Test your function here
    df_lm_aggregate = lt.aggregate_rows(df_fine, df_coarse, model="sentence-transformers/all-mpnet-base-v2", left_on="Fine Category Name", right_on="Coarse Category Name")
    assert isinstance(df_lm_aggregate, pd.DataFrame)
    # Add more assertions to check the correctness of the output

def test_lt_merge_multi_columns():
    df1 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_multi_1.csv"))
    df2 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_multi_2.csv"))

    # Test your function here
    df_lm_matched = lt.merge(df2, df1, merge_type='1:m', on=["CompanyName", "ProductDescription"], model="sentence-transformers/all-MiniLM-L6-v2", left_on=None, right_on=None)
    assert isinstance(df_lm_matched, pd.DataFrame)
    # Add more assertions to check the correctness of the output

def test_lt_merge_blocking():
    df1 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_comp_1.csv"))
    df2 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_comp_2.csv"))

    # Test your function here
    df_lm_matched = lt.merge_blocking(df2, df1, merge_type='1:m', on="CompanyName", model="sentence-transformers/all-MiniLM-L6-v2", left_on=None, right_on=None, blocking_vars=["Country"])
    assert isinstance(df_lm_matched, pd.DataFrame)
    # Add more assertions to check the correctness of the output

# Add more test functions for other functionalities of your code

# French to English Translation Test
def test_lt_merge_crosslingual_french_english():
    df_french = pd.read_csv(os.path.join(DATA_DIR_PATH, "translation_1.csv"))
    df_english = pd.read_csv(os.path.join(DATA_DIR_PATH, "translation_2.csv"))

    # Test your function here
    df_lm_matched = lt.merge(df_french, df_english, merge_type='1:m', left_on="Libellé du Produit", right_on="Product Label", model="distiluse-base-multilingual-cased-v1")
    assert isinstance(df_lm_matched, pd.DataFrame)
    # Add more assertions to check the correctness of the output


###Test deduplication
def test_lt_dedup_rows():
    df=pd.read_csv(os.path.join(DATA_DIR_PATH,"toy_comp_2.csv"))
    df_dedup=lt.dedup_rows(df,on="CompanyName",model="sentence-transformers/all-MiniLM-L6-v2",cluster_type= "agglomerative",
        cluster_params= {'threshold': 0.7})
    assert isinstance(df_dedup, pd.DataFrame,)
    # Add more assertions to check the correctness of the output


##Test clustering
def test_lt_cluster_rows():
    df=pd.read_csv(os.path.join(DATA_DIR_PATH,"toy_comp_2.csv"))
    df_cluster=lt.cluster_rows(df,on="CompanyName",model="sentence-transformers/all-MiniLM-L6-v2",cluster_type= "agglomerative",
        cluster_params= {'threshold': 0.7})
    assert isinstance(df_cluster, pd.DataFrame,)
    # Add more assertions to check the correctness of the output

def test_lt_cluster_rows_noargs():
    df=pd.read_csv(os.path.join(DATA_DIR_PATH,"toy_comp_2.csv"))
    df_cluster=lt.cluster_rows(df,on="CompanyName",model="sentence-transformers/all-MiniLM-L6-v2")
    assert isinstance(df_cluster, pd.DataFrame)

def test_lt_cluster_rows_slink():
    df=pd.read_csv(os.path.join(DATA_DIR_PATH,"toy_comp_2.csv"))
    df_cluster=lt.cluster_rows(df,on="CompanyName",model="sentence-transformers/all-MiniLM-L6-v2",cluster_type= "SLINK")
    assert isinstance(df_cluster, pd.DataFrame)

def test_lt_cluster_rows_hdbscan():
    df=pd.read_csv(os.path.join(DATA_DIR_PATH,"toy_comp_2.csv"))
    df_cluster=lt.cluster_rows(df,on="CompanyName",model="sentence-transformers/all-MiniLM-L6-v2",cluster_type= "HDBScan")
    assert isinstance(df_cluster, pd.DataFrame)

def test_lt_cluster_rows_hdbscan_params():
    df=pd.read_csv(os.path.join(DATA_DIR_PATH,"toy_comp_2.csv"))
    df_cluster=lt.cluster_rows(df,on="CompanyName",model="sentence-transformers/all-MiniLM-L6-v2",cluster_type= "HDBScan",
        cluster_params= {'min cluster size': 2, 'min samples': 1})
    assert isinstance(df_cluster, pd.DataFrame)

def test_lt_cluster_rows_slink_params():
    df=pd.read_csv(os.path.join(DATA_DIR_PATH,"toy_comp_2.csv"))
    df_cluster=lt.cluster_rows(df,on="CompanyName",model="sentence-transformers/all-MiniLM-L6-v2",cluster_type= "SLINK",
        cluster_params= {'min cluster size': 2, 'threshold': 0.1, 'metric': 'cosine'})
    assert isinstance(df_cluster, pd.DataFrame)

def test_lt_cluster_rows_agglomerative_params():
    df=pd.read_csv(os.path.join(DATA_DIR_PATH,"toy_comp_2.csv"))
    df_cluster=lt.cluster_rows(df,on="CompanyName",model="sentence-transformers/all-MiniLM-L6-v2",cluster_type= "agglomerative",
        cluster_params= {'threshold': 0.7, 'clustering linkage': 'average', 'metric': 'cosine'})
    assert isinstance(df_cluster, pd.DataFrame)


###Test evaluate pairs
def test_lt_evaluate_pairs():
    df=pd.read_csv(os.path.join(DATA_DIR_PATH,"toy_pairs.csv"))
    df_eval=lt.evaluate_pairs(df, model="sentence-transformers/all-MiniLM-L6-v2",left_on="company_name_1",right_on="company_name_2",openai_key=None)

    assert isinstance(df_eval,pd.DataFrame)

##Test pairwise evaluate
def test_lt_all_pair_combos_evaluate():
    df=pd.read_csv(os.path.join(DATA_DIR_PATH,"toy_pairs.csv"))
    df_eval=lt.all_pair_combos_evaluate(df, model="sentence-transformers/all-MiniLM-L6-v2",left_on="company_name_1",right_on="company_name_2",openai_key=None)
    print(df_eval)
    assert isinstance(df_eval,pd.DataFrame)

##Test knn
def test_lt_merge_knn():
    df1 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_comp_1.csv"))
    df2 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_comp_2.csv"))

    # Test your function here
    df_lm_matched = lt.merge_knn(df2, df1, on="CompanyName", model="sentence-transformers/all-MiniLM-L6-v2", left_on=None, right_on=None,
                                 k=1)
    print(df_lm_matched)
    assert isinstance(df_lm_matched, pd.DataFrame)

    ###Check if identical to merge
    df_lm_matched2 = lt.merge(df2, df1, merge_type='1:m', on="CompanyName", model="sentence-transformers/all-MiniLM-L6-v2", left_on=None, right_on=None)

    ###Check if k=2 is different
    df_lm_matched3_2nn = lt.merge_knn(df2, df1, on="CompanyName", model="sentence-transformers/all-MiniLM-L6-v2", left_on=None, right_on=None,
                                    k=2)
    
    print(df_lm_matched3_2nn)

    ###Asser that length is double
    assert len(df_lm_matched3_2nn) == len(df_lm_matched)*2


    assert df_lm_matched.equals(df_lm_matched2)


def test_lt_merge_knn_suffixes():
    df1 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_comp_1.csv"))
    df2 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_comp_2.csv"))

    # Test your function here
    df_lm_matched = lt.merge_knn(df2, df1, on="CompanyName", model="sentence-transformers/all-MiniLM-L6-v2", left_on=None, right_on=None,
                                 k=1,suffixes=("_1","_2"))
    print(df_lm_matched)
    assert isinstance(df_lm_matched, pd.DataFrame)

    ###Check if identical to merge
    df_lm_matched2 = lt.merge(df2, df1, on="CompanyName", model="sentence-transformers/all-MiniLM-L6-v2", left_on=None, right_on=None,suffixes=("_1","_2"))

    ###Check if k=2 is different
    df_lm_matched3_2nn = lt.merge_knn(df2, df1, on="CompanyName", model="sentence-transformers/all-MiniLM-L6-v2", left_on=None, right_on=None,
                                    k=2,suffixes=("_1","_2"))
    
    print(df_lm_matched3_2nn)

    ###Asser that length is double
    assert len(df_lm_matched3_2nn) == len(df_lm_matched)*2
    
    ##Check if suffixes are added
    print(df_lm_matched3_2nn.columns)
    assert "CompanyName_1" in df_lm_matched3_2nn.columns
    assert "CompanyName_2" in df_lm_matched3_2nn.columns


    assert df_lm_matched.equals(df_lm_matched2)


def test_lt_merge_range_many_to_many(monkeypatch):
    df1 = pd.DataFrame({"CompanyName": ["alpha", "beta"]})
    df2 = pd.DataFrame({"CompanyName": ["alpha inc", "alpha llc", "beta corp"]})

    embedding_map = {
        "alpha": np.array([1.0, 0.0], dtype=np.float32),
        "beta": np.array([0.0, 1.0], dtype=np.float32),
        "alpha inc": np.array([1.0, 0.0], dtype=np.float32),
        "alpha llc": np.array([0.8, 0.2], dtype=np.float32),
        "beta corp": np.array([0.0, 1.0], dtype=np.float32),
    }

    def _fake_infer_embeddings(strings, model, **kwargs):
        return np.vstack([embedding_map[item] for item in strings]).astype(np.float32)

    monkeypatch.setattr(infer_mod, "infer_embeddings", _fake_infer_embeddings)

    out = lt.merge_range(
        df1,
        df2,
        on="CompanyName",
        model=object(),
        sim_threshold=0.01,
    )
    
    print(out)

    assert isinstance(out, pd.DataFrame)
    assert "score" in out.columns
    assert (out["score"] >= 0.01).all()


def test_lt_merge_range_retains_unmatched_left_rows():
    df1 = pd.DataFrame({"CompanyName": ["alpha","beta", "gamma"]})
    df2 = pd.DataFrame({"CompanyName": ["beta_corp", "delta inc","alpha llc","epsilon llc"]})


    out = lt.merge_range(
        df1,
        df2,
        on="CompanyName",
        model="sentence-transformers/all-MiniLM-L6-v2",
        sim_threshold=0.01,
    )

    print(out)
    assert isinstance(out, pd.DataFrame)
    assert "score" in out.columns


def test_lt_merge_invalid_merge_type_raises():
    df1 = pd.DataFrame({"name": ["a"]})
    df2 = pd.DataFrame({"name": ["b"]})

    with pytest.raises(ValueError, match="Invalid merge type"):
        lt.merge(df1, df2, merge_type="bad", on="name", model=object())


def test_lt_merge_rejects_existing_id_lt():
    df1 = pd.DataFrame({"name": ["a"], "id_lt": [0]})
    df2 = pd.DataFrame({"name": ["b"]})

    with pytest.raises(ValueError, match="id_lt already exists"):
        lt.merge(df1, df2, on="name", model=object())


def test_lt_merge_on_none_uses_left_shared_column_order(monkeypatch):
    df1 = pd.DataFrame({"left_only": ["l"], "b": ["b1"], "a": ["a1"]})
    df2 = pd.DataFrame({"a": ["a2"], "b": ["b2"], "right_only": ["r"]})

    captured = {"left": None, "right": None}

    def _fake_serialize(df, columns, **kwargs):
        key = "left" if "left_only" in df.columns else "right"
        captured[key] = columns
        return [" ".join([str(df.iloc[0][col]) for col in columns])]

    def _fake_embed(strings, model, **kwargs):
        return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(infer_mod, "serialize_columns", _fake_serialize)
    monkeypatch.setattr(infer_mod, "infer_embeddings", _fake_embed)

    out = lt.merge(df1, df2, on=None, model=object())

    assert isinstance(out, pd.DataFrame)
    assert captured["left"] == ["b", "a"]
    assert captured["right"] == ["b", "a"]


def test_lt_merge_blocking_unmatched_rows_have_uniform_schema_and_status(monkeypatch):
    df1 = pd.DataFrame(
        {
            "name": ["match_left", "left_only"],
            "country": ["US", "CA"],
        }
    )
    df2 = pd.DataFrame(
        {
            "name": ["match_right", "right_only"],
            "country": ["US", "MX"],
        }
    )

    def _fake_merge(df1, df2, **kwargs):
        assert len(df1) == 1
        assert len(df2) == 1
        return pd.DataFrame(
            {
                "name_x": [df1.iloc[0]["name"]],
                "country_x": [df1.iloc[0]["country"]],
                "id_lt_x": [0],
                "name_y": [df2.iloc[0]["name"]],
                "country_y": [df2.iloc[0]["country"]],
                "id_lt_y": [0],
                "score": [0.99],
            }
        )

    monkeypatch.setattr(infer_mod, "merge", _fake_merge)

    out = lt.merge_blocking(
        df1,
        df2,
        on="name",
        model=object(),
        blocking_vars=["country"],
    )

    assert "match_status" in out.columns
    assert set(out["match_status"].dropna().unique()) == {"matched_block", "left_only_block", "right_only_block"}
    assert {"name_x", "country_x", "name_y", "country_y", "score"}.issubset(set(out.columns))
    left_only = out[out["match_status"] == "left_only_block"].iloc[0]
    right_only = out[out["match_status"] == "right_only_block"].iloc[0]
    assert pd.isna(left_only["name_y"])
    assert pd.isna(right_only["name_x"])


def test_lt_transform_rows_preserves_null_rows(monkeypatch):
    df = pd.DataFrame({"text": ["alpha", None, "beta"]})

    class _FakeClient:
        pass

    def _fake_transform_column(df, column, transform_fn, fn_kwargs, **kwargs):
        out_col = kwargs["output_column"]
        out = df.copy()
        out[out_col] = [f"ok::{x}" for x in out[column].tolist()]
        return out

    monkeypatch.setattr(infer_mod.openai, "OpenAI", lambda **kwargs: _FakeClient())
    monkeypatch.setattr(infer_mod, "transform_column", _fake_transform_column)

    out = lt.transform_rows(
        df,
        on="text",
        openai_key="dummy",
        model="gpt-4o-mini",
        progress_bar=False,
    )

    assert len(out) == 3
    assert "transformed_text" in out.columns
    assert out.loc[0, "transformed_text"] == "ok::alpha"
    assert pd.isna(out.loc[1, "transformed_text"])
    assert out.loc[2, "transformed_text"] == "ok::beta"


def test_lt_transform_rows_requires_on_column():
    with pytest.raises(ValueError, match="Must specify `on=`"):
        lt.transform_rows(pd.DataFrame({"a": [1]}), on=None, openai_key="dummy")


def test_lt_classify_rows_openai_requires_prompt_or_topic():
    with pytest.raises(ValueError, match="openai_topic or openai_prompt"):
        lt.classify_rows(
            pd.DataFrame({"text": ["hello"]}),
            on="text",
            model="gpt-4o-mini",
            openai_key="dummy",
        )


def test_lt_classify_rows_api_forwards_batch_and_throttle(monkeypatch):
    df = pd.DataFrame({"text": ["a", "b", "c"]})
    captured = {}

    def _fake_predict_rows_with_llm(strings_col, model, **kwargs):
        captured["strings_col"] = strings_col
        captured["model"] = model
        captured.update(kwargs)
        return [1, 0, 1]

    monkeypatch.setattr(infer_mod, "predict_rows_with_llm", _fake_predict_rows_with_llm)

    out = lt.classify_rows(
        df,
        on="text",
        model="gpt-4o-mini",
        openai_key="dummy",
        openai_prompt="Binary classify",
        batch_size=2,
        api_throttle_seconds=0.4,
        api_workers=3,
        progress_bar=False,
    )

    assert out["clf_preds_text"].tolist() == [1, 0, 1]
    assert captured["strings_col"] == ["a", "b", "c"]
    assert captured["batch_size"] == 2
    assert captured["throttle_seconds"] == 0.4
    assert captured["workers"] == 3
    assert captured["progress_bar"] is False


def test_lt_transform_rows_gemini_routes_to_gemini_transform(monkeypatch):
    df = pd.DataFrame({"text": ["alpha"]})
    captured = {}

    class _FakeClient:
        def __init__(self, api_key, http_options=None):
            captured["api_key"] = api_key

    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = _FakeClient
    fake_genai.types = types.SimpleNamespace(HttpOptions=lambda **kwargs: kwargs)

    def _fake_transform_column(df, column, transform_fn, fn_kwargs, **kwargs):
        captured["transform_fn"] = transform_fn
        captured["model_name"] = fn_kwargs["model"]
        captured["client"] = fn_kwargs["gemini_client"]
        captured["throttle_seconds"] = kwargs.get("throttle_seconds")
        out_col = kwargs["output_column"]
        out = df.copy()
        out[out_col] = ["gemini::ok"]
        return out

    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setattr(infer_mod, "transform_column", _fake_transform_column)

    out = lt.transform_rows(
        df,
        on="text",
        gemini_key="g-key",
        llm_provider="gemini",
        model="gemini-3.5-flash-lite",
        openai_prompt="Normalize",
        batch_size=4,
        api_throttle_seconds=0.2,
        progress_bar=False,
    )

    assert out["transformed_text"].tolist() == ["gemini::ok"]
    assert captured["api_key"] == "g-key"
    assert captured["model_name"] == "gemini-3.5-flash-lite"
    assert captured["transform_fn"] is infer_mod.gemini_transform
    assert captured["throttle_seconds"] == 0.2

# def test_knn_range_search():
#     df1 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_comp_1.csv"))
#     df2 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_comp_2.csv"))
#     df_lm_matched_range_search = lt.merge_knn(df2, df1, on="CompanyName", model="sentence-transformers/all-MiniLM-L6-v2", left_on=None, right_on=None,
#                                     drop_sim_threshold=0.5,suffixes=("_1","_2"),use_range_search=True)
#     df_lm_matched_2nn = lt.merge_knn(df2, df1, on="CompanyName", model="sentence-transformers/all-MiniLM-L6-v2", left_on=None, right_on=None,
#                                     k=2,suffixes=("_1","_2"))
    
#     print(df_lm_matched_range_search.shape)
#     print(df_lm_matched_2nn.shape)

def test_lt_classify_rows_single_col_binary():
    df = pd.read_csv(os.path.join(DATA_DIR_PATH, "protests_toy_sample_binary.csv"))

    df_clf_output = lt.classify_rows(df, on="article", model="distilroberta-base")
    assert isinstance(df_clf_output, pd.DataFrame)
    assert "clf_preds_article" in df_clf_output
    assert df_clf_output["clf_preds_article"].isin([0, 1]).all()

    print(df_clf_output)


def test_lt_classify_rows_multi_col_binary():
    df = pd.read_csv(os.path.join(DATA_DIR_PATH, "protests_toy_sample_binary.csv"))

    df_clf_output = lt.classify_rows(df, on=["article", "image_id"], model="distilroberta-base")
    assert isinstance(df_clf_output, pd.DataFrame)
    assert "clf_preds_article-image_id" in df_clf_output
    assert df_clf_output["clf_preds_article-image_id"].isin([0, 1]).all()

    print(df_clf_output)


def test_lt_classify_rows_single_col_ternary():
    df = pd.read_csv(os.path.join(DATA_DIR_PATH, "protests_toy_sample_ternary.csv"))

    df_clf_output = lt.classify_rows(df, on="article", model="distilroberta-base", num_labels=3)
    assert isinstance(df_clf_output, pd.DataFrame)
    assert "clf_preds_article" in df_clf_output
    assert df_clf_output["clf_preds_article"].isin([0, 1, 2]).all()

    print(df_clf_output)


def test_lt_classify_rows_multi_col_ternary():
    df = pd.read_csv(os.path.join(DATA_DIR_PATH, "protests_toy_sample_ternary.csv"))

    df_clf_output = lt.classify_rows(df, on=["article", "image_id"], model="distilroberta-base", num_labels=3)
    assert isinstance(df_clf_output, pd.DataFrame)
    assert "clf_preds_article-image_id" in df_clf_output
    # assert df_clf_output["clf_preds_article-image_id"].isin([0, 1, 2]).all()

    print(df_clf_output)


@pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="OpenAI API keys not found in environment variable")
def test_lt_classify_rows_single_col_binary_openai():
    df = pd.read_csv(os.path.join(DATA_DIR_PATH, "protests_toy_sample_binary.csv"))

    df_clf_output = lt.classify_rows(df, on="article", model="gpt-3.5-turbo", num_labels=2,
                                     openai_key=os.getenv("OPENAI_API_KEY"), openai_topic="protests")
    assert isinstance(df_clf_output, pd.DataFrame)
    assert "clf_preds_article" in df_clf_output
    assert df_clf_output["clf_preds_article"].isin([0, 1]).all()

    print(df_clf_output)


@pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="OpenAI API keys not found in environment variable")
def test_lt_classify_rows_multi_col_ternary_openai():
    df = pd.read_csv(os.path.join(DATA_DIR_PATH, "protests_toy_sample_ternary.csv"))

    label_dict = {"Protest": 1, "Riot": 2, "Neither": 0}
    openai_prompt = "Determine whether the text is about protests, riots or neither. Protest/Riot/Neither: "
    df_clf_output = lt.classify_rows(df, on=["article", "image_id"], model="gpt-4o-mini", num_labels=3,
                                     openai_key=os.getenv("OPENAI_API_KEY"),
                                     openai_prompt=openai_prompt, label_map=label_dict)
    print(df_clf_output)
    assert isinstance(df_clf_output, pd.DataFrame)
    assert "clf_preds_article-image_id" in df_clf_output
    # assert df_clf_output["clf_preds_article-image_id"].isin([0, 1, 2]).all()

    print(df_clf_output)
    
##Test a merge using openai
@pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="OpenAI API keys not found in environment variable")
def test_lt_merge_openai_embeddings():
    df1 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_comp_1.csv"))
    df2 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_comp_2.csv"))

    # Test your function here
    df_lm_matched = lt.merge(df2, df1, merge_type='1:m', on=["CompanyName","Country"], 
                             model="text-embedding-ada-002", left_on=None, right_on=None
                             ,openai_key=os.getenv("OPENAI_API_KEY"))
    print(df_lm_matched)
    assert isinstance(df_lm_matched, pd.DataFrame)
    
import os
import pytest
import pandas as pd
import linktransformer as lt

@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="OpenAI API key not found in environment"
)
def test_lt_transform_rows_multi_col_openai():
    # prepare toy DataFrame with obvious spelling mistakes
    df = pd.DataFrame({
        "text": ["Ths is a tst.", "Anothr exmple."],
        "suffix": [" corrctn", " pls fix"]
    })

    # call transform_rows on both columns
    df_out = lt.transform_rows(
        df,
        on=["text", "suffix"],
        openai_key=os.getenv("OPENAI_API_KEY"),
        batch_size=2,
        model="gpt-4o",
        # explicit prompt to fix spelling mistakes
        openai_prompt=(
            "Fix spelling mistakes in each of the following strings. "
            "Return a JSON array of the corrected strings."
        )
    )
    
    print(df_out)

    # basic sanity checks
    assert isinstance(df_out, pd.DataFrame)
    # the new column should combine the two input column names
    assert "transformed_text-suffix" in df_out.columns

    corrected = df_out["transformed_text-suffix"].tolist()
    # length must match original DataFrame
    assert len(corrected) == len(df)

    # each corrected entry should be a non-empty string
    for entry in corrected:
        assert isinstance(entry, str)
        assert entry.strip() != ""

@pytest.mark.skipif("GEMINI_API_KEY" not in os.environ, reason="Gemini API key not found in environment")
def test_lt_merge_gemini_live():
    pytest.importorskip("google.genai")

    df1 = pd.DataFrame(
        {
            "CompanyName": ["Tech Corporation", "InfoTech Solutions", "AlphaSoft Systems"],
            "Country": ["USA", "USA", "Canada"],
        }
    )
    df2 = pd.DataFrame(
        {
            "CompanyName": ["Tech Corp", "InfoTech Soln", "AlphaSoft"],
            "Country": ["USA", "USA", "Canada"],
        }
    )

    df_lm_matched = lt.merge(
        df1,
        df2,
        on=["CompanyName", "Country"],
        model="gemini-embedding-001",
        gemini_key=os.getenv("GEMINI_API_KEY"),
    )

    assert isinstance(df_lm_matched, pd.DataFrame)
    assert "score" in df_lm_matched.columns
    assert len(df_lm_matched) == len(df1)


if __name__ == "__main__":
    # test_lt_data_dir_path()
    # test_lt_merge()
    # test_lt_aggregate_rows()
    # test_lt_merge_multi_columns()
    # test_lt_merge_blocking()
    # test_lt_merge_crosslingual_french_english()
    # test_lt_dedup_rows()
    # test_lt_merge_knn()
    # # test classification
    # test_lt_classify_rows_single_col_binary()
    # test_lt_classify_rows_multi_col_binary()
    
    ##OpenAI tests
    test_lt_merge_openai_embeddings()
    test_lt_classify_rows_single_col_binary_openai()
    test_lt_classify_rows_multi_col_ternary_openai()
    test_lt_transform_rows_multi_col_openai()
    
    # Gemini test
    test_lt_merge_gemini_live()
    
