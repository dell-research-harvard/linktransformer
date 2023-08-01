import os
import pandas as pd
from linktransformer.data import DATA_DIR_PATH
from linktransformer.infer import lm_merge, lm_aggregate, lm_merge_blocking, lm_aggregate

def test_lm_merge():
    df1 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_comp_1.csv"))
    df2 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_comp_2.csv"))

    # Test your function here
    df_lm_matched = lm_merge(df2, df1, merge_type='1:m', on="CompanyName", model="all-MiniLM-L6-v2", left_on=None, right_on=None)
    assert isinstance(df_lm_matched, pd.DataFrame)
    # Add more assertions to check the correctness of the output

def test_lm_aggregate():
    df_coarse = pd.read_csv(os.path.join(DATA_DIR_PATH, "coarse.csv"))
    df_fine = pd.read_csv(os.path.join(DATA_DIR_PATH, "fine.csv"))

    # Test your function here
    df_lm_aggregate = lm_aggregate(df_fine, df_coarse, model="all-mpnet-base-v2", left_on="Fine Category Name", right_on="Coarse Category Name")
    assert isinstance(df_lm_aggregate, pd.DataFrame)
    # Add more assertions to check the correctness of the output

def test_lm_merge_with_multiple_columns():
    df1 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_multi_1.csv"))
    df2 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_multi_2.csv"))

    # Test your function here
    df_lm_matched = lm_merge(df2, df1, merge_type='1:m', on=["CompanyName", "ProductDescription"], model="all-MiniLM-L6-v2", left_on=None, right_on=None)
    assert isinstance(df_lm_matched, pd.DataFrame)
    # Add more assertions to check the correctness of the output

def test_lm_merge_with_blocking_df():
    df1 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_comp_1.csv"))
    df2 = pd.read_csv(os.path.join(DATA_DIR_PATH, "toy_comp_2.csv"))

    # Test your function here
    df_lm_matched = lm_merge_blocking(df2, df1, merge_type='1:m', on="CompanyName", model="all-MiniLM-L6-v2", left_on=None, right_on=None, blocking_vars=["Country"])
    assert isinstance(df_lm_matched, pd.DataFrame)
    # Add more assertions to check the correctness of the output

# Add more test functions for other functionalities of your code

# French to English Translation Test
def test_french_to_english_translation():
    df_french = pd.read_csv(os.path.join(DATA_DIR_PATH, "translation_1.csv"))
    df_english = pd.read_csv(os.path.join(DATA_DIR_PATH, "translation_2.csv"))

    # Test your function here
    df_lm_matched = lm_merge(df_french, df_english, merge_type='1:m', left_on="Libellé du Produit", right_on="Product Label", model="distiluse-base-multilingual-cased-v1")
    assert isinstance(df_lm_matched, pd.DataFrame)
    # Add more assertions to check the correctness of the output


if __name__ == "__main__":
    test_lm_merge()
    test_lm_aggregate()
    test_lm_merge_with_multiple_columns()
    test_lm_merge_with_blocking_df()
    test_french_to_english_translation()