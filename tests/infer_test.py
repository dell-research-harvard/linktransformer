import os
import pandas as pd
from linktransformer import DATA_DIR_PATH
import linktransformer as lt

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
    df_clf_output = lt.classify_rows(df, on=["article", "image_id"], model="gpt-3.5-turbo", num_labels=3,
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
    pytest.importorskip("google.generativeai")

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
    
