###Inference and Linkage script
###We want to link dfs together using embeddings
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
import faiss
from typing import Union, List, Optional, Tuple,Dict, Any
from pandas import DataFrame

from linktransformer.cluster_fns import cluster
# from linktransformer.utils import serialize_columns, infer_embeddings, load_model, load_clf, cosine_similarity_corresponding_pairs, tokenize_data_for_inference, predict_rows_with_openai
from linktransformer.utils import *
from linktransformer.utils import _is_gemini_embedding_model
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from transformers import TrainingArguments, Trainer



def merge(
    df1: DataFrame,
    df2: DataFrame,
    merge_type: str = None,
    on: Optional[Union[str, List[str]]] = None,
    model: Union[str, LinkTransformer] = "all-MiniLM-L6-v2",
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    suffixes: Tuple[str, str] = ("_x", "_y"),
    batch_size: int = 128,
    openai_key: Optional[str] = None,
    gemini_key: Optional[str] = None,
) -> DataFrame:
    """
    Merge two dataframes using language model embeddings.

    :param df1 (DataFrame): First dataframe (left).
    :param df2 (DataFrame): Second dataframe (right).
    :param merge_type (str): Type of merge to perform (1:m or m:1 or 1:1).
        .. deprecated:: 0.1.14
        No longer useful as it only validates whether the join columns are unique; it will be removed in the future.
    :param model (str): Language model to use.
    :param on (Union[str, List[str]], optional): Column(s) to join on in df1 and df2. Defaults to None.
    :param left_on (Union[str, List[str]], optional): Column(s) to join on in df1. Defaults to None.
    :param right_on (Union[str, List[str]], optional): Column(s) to join on in df2. Defaults to None.
    :param suffixes (Tuple[str, str]): Suffixes to use for overlapping columns. Defaults to ('_x', '_y').
    :param batch_size (int): Batch size for inferencing embeddings. Defaults to 128.
    :param openai_key (str, optional): OpenAI API key for OpenAI models. Defaults to None.
    :param gemini_key (str, optional): Gemini API key for Gemini embedding models. Defaults to None.
    :return: DataFrame: The merged dataframe.
    """
    ## Set common columns as on if not specified
    if on is None:
        on = list(set(df1.columns).intersection(set(df2.columns)))

    ## If left_on or right_on is not specified, set it to on
    if left_on is None:
        left_on = on
    if right_on is None:
        right_on = on

    on = None

    ##Deprication warning for merge_type
    if merge_type is not None:
        warnings.warn("merge_type is deprecated. It will be removed in the future as it only validates whether the join columns are unique", DeprecationWarning)
    
    ### Check if how is valid
    if merge_type not in ["1:m", "m:1", "1:1", None]:
        raise ValueError(f"Invalid merge type: {merge_type}")

    if merge_type == "1:m":
        if df1[left_on].duplicated().any():
           print("Warning: Keys in df1 are not unique")

    if merge_type == "m:1":
        ## Check if keys in df2 are unique
        if df2[right_on].duplicated().any():
            print("Warning: Keys in df2 are not unique")
    if merge_type == "1:1":
        ## Check if keys in df1 are unique
        if df1[left_on].duplicated().any():
           print("Warning: Keys in df1 are not unique")
        ## Check if keys in df2 are unique
        if df2[right_on].duplicated().any():
           print("Warning: Keys in df1 are not unique")

    df1 = df1.copy()
    df2 = df2.copy()
    ## give ids to each df
    ##Ensure that there is no id_lt column in df1 or df2
    if "id_lt" in df1.columns:
        raise ValueError(f"Column id_lt already exists in df1, please rename it to proceed")
    if "id_lt" in df2.columns:
        raise ValueError(f"Column id_lt already exists in df2,please rename it to proceed")

    df1.loc[:, "id_lt"] = np.arange(len(df1))
    df2.loc[:, "id_lt"] = np.arange(len(df2))

    ## Load the model if string
    if isinstance(model, str):
        if openai_key is None and gemini_key is None and not _is_gemini_embedding_model(model):
            model = load_model(model)


    model_is_api_embedding = isinstance(model, str) and (
        _is_openai_embedding_model(model) or _is_gemini_embedding_model(model)
    )

    if isinstance(right_on, list):
        if model_is_api_embedding:
            strings_right = serialize_columns(df2, right_on, sep_token="<SEP>")
        else:
            strings_right = serialize_columns(df2, right_on, model=model)
    if isinstance(left_on, list):
        if model_is_api_embedding:
            strings_left = serialize_columns(df1, left_on, sep_token="<SEP>")
        else:
            strings_left = serialize_columns(df1, left_on, model=model)
    else:
        strings_left = df1[left_on].tolist()
        strings_right = df2[right_on].tolist()



    ## Infer embeddings for df1
    embeddings1 = infer_embeddings(strings_left, model, batch_size=batch_size, openai_key=openai_key, gemini_key=gemini_key)
    ## Infer embeddings for df2
    embeddings2 = infer_embeddings(strings_right, model, batch_size=batch_size, openai_key=openai_key, gemini_key=gemini_key)

    ### Expand dim if embeddings are 1d (numpy)
    if len(embeddings1.shape) == 1:
        embeddings1 = np.expand_dims(embeddings1, axis=0)
    if len(embeddings2.shape) == 1:
        embeddings2 = np.expand_dims(embeddings2, axis=0)
    ## Normalize embedding tensors using numpy

    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
    ## Create index
    index = faiss.IndexFlatIP(embeddings1.shape[1])

    ## Add to index depending on merge type
    index.add(embeddings2)


    ## Search index
    D, I = index.search(embeddings1, 1)
  

    ## Check nearest neighbor of the first text in df1 as a test
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    ## Fuzzily merge the dfs based on the faiss index queries
    df_lm_matched = df1.merge(df2.iloc[I.flatten()].reset_index(drop=True), left_index=True, right_index=True, how="inner",suffixes=suffixes)
    ### Add score column
    df_lm_matched["score"] = D.flatten()

        
    return df_lm_matched

    


def merge_blocking(
    df1: DataFrame,
    df2: DataFrame,
    merge_type: str = None,
    on: Optional[Union[str, List[str]]] = None,
    model: Union[str, LinkTransformer] = "all-MiniLM-L6-v2",
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    blocking_vars: Optional[List[str]] = None,
    suffixes: Tuple[str, str] = ("_x", "_y"),
    batch_size: int = 128,
    openai_key: Optional[str] = None,
    gemini_key: Optional[str] = None,
) -> DataFrame:
    """
    Merge two dataframes using language model embeddings with optional blocking.

    :param df1 (DataFrame): First dataframe (left).
    :param df2 (DataFrame): Second dataframe (right).
    :param merge_type (str): Type of merge to perform (1:m or m:1 or 1:1).
        .. deprecated:: 0.1.14
        No longer useful as it only validates whether the join columns are unique; it will be removed in the future.
    :param model (str): Language model to use.
    :param on (Union[str, List[str]], optional): Column(s) to join on in df1 and df2. Defaults to None.
    :param left_on (Union[str, List[str]], optional): Column(s) to join on in df1. Defaults to None.
    :param right_on (Union[str, List[str]], optional): Column(s) to join on in df2. Defaults to None.
    :param blocking_vars (List[str], optional): Columns to use for blocking. Defaults to None.
    :param suffixes (Tuple[str, str]): Suffixes to use for overlapping columns. Defaults to ('_x', '_y').
    :param batch_size (int): Batch size for inferencing embeddings. Defaults to 128.
    :param openai_key (str, optional): OpenAI API key for OpenAI models. Defaults to None.
    :param gemini_key (str, optional): Gemini API key for Gemini embedding models. Defaults to None.
    :return: DataFrame: The merged dataframe.
    """
    ### For blocking, we need to chunk the dfs into blocks
    ### First, we need to check if blocking vars are specified
    if blocking_vars is None:
        print("No blocking vars specified, matching between all rows")
        df_lm_matched = merge(df1, df2, merge_type=merge_type, on=on, model=model, left_on=left_on,
                                    right_on=right_on, suffixes=suffixes, batch_size=batch_size,
                                     openai_key=openai_key, gemini_key=gemini_key)
        return df_lm_matched
    else:
        ## Partition the dfs into blocks
        ## First, check if blocking vars are in df1 and df2 - both should exist
        if not set(blocking_vars).issubset(set(df1.columns)):
            raise ValueError(f"Blocking vars {blocking_vars} not in df1")
        if not set(blocking_vars).issubset(set(df2.columns)):
            raise ValueError(f"Blocking vars {blocking_vars} not in df2")

        ## Now, chunk the dfs into blocks
        ## First, get the number of blocks
        groupby_key = blocking_vars[0] if len(blocking_vars) == 1 else blocking_vars
        num_blocks_1 = len(df1.groupby(groupby_key))
        num_blocks_2 = len(df2.groupby(groupby_key))
        print(f"Number of blocks in df1: {num_blocks_1}")
        print(f"Number of blocks in df2: {num_blocks_2}")

        ## Now, get the blocks
        df1_blocks = df1.groupby(groupby_key)
        df2_blocks = df2.groupby(groupby_key)

        ## Now, we need to merge each block in df1 with each block in df2.
        ## We need to keep track of the blocks that have been merged

        ## First, create a list of all the blocks in df1 and df2
        df1_block_list = list(df1_blocks.groups.keys())
        df2_block_list = list(df2_blocks.groups.keys())

        ### Iterate through each block in df1. Check if the block is in df2. If it is, merge the blocks.
        ### If not, add keys to skipped keys list
        merged_dfs = []
        skipped_keys_1 = set(df1_block_list).difference(set(df2_block_list))
        skipped_keys_2 = set(df2_block_list).difference(set(df1_block_list))
        common_keys = set(df1_block_list).intersection(set(df2_block_list))

        ## Load the model if string
        if isinstance(model, str):
            if openai_key is None and gemini_key is None and not _is_gemini_embedding_model(model):
                model = load_model(model)      

        for block_1 in common_keys:
            print(f"Merging block {block_1}")
            df1_block = df1_blocks.get_group(block_1)
            df2_block = df2_blocks.get_group(block_1)
            ## Merge the blocks
            df_block_matched = merge(df1_block, df2_block, merge_type=merge_type, on=on, model=model,
                                           left_on=left_on, right_on=right_on, suffixes=suffixes,
                                           batch_size=batch_size, openai_key=openai_key, gemini_key=gemini_key)
            ## Add to merged dfs
            merged_dfs.append(df_block_matched)

        ## Add df corresponding to skipped keys
        for block_1 in skipped_keys_1:
            df1_block = df1_blocks.get_group(block_1)
            merged_dfs.append(df1_block)

        for block_2 in skipped_keys_2:
            df2_block = df2_blocks.get_group(block_2)
            merged_dfs.append(df2_block)

        ## Concatenate the merged dfs
        df_lm_matched = pd.concat(merged_dfs, axis=0).reset_index(drop=True)
        
        
        
        return df_lm_matched




def aggregate_rows(
    df: DataFrame,
    ref_df: DataFrame,
    model: Union[str, LinkTransformer] = "all-MiniLM-L6-v2",
    left_on: Union[str, List[str]]=None,
    right_on: Union[str, List[str]]=None,
    openai_key: str = None,
    gemini_key: str = None,
) -> DataFrame:
    """
    Aggregate the dataframe based on a reference dataframe using a language model.

    :param df (DataFrame): Dataframe to aggregate.
    :param ref_df (DataFrame): Reference dataframe to aggregate on.
    :param model (str): Language model to use.
    :param left_on (Union[str, List[str]]): Column(s) to aggregate on in df.
    :param right_on (Union[str, List[str]]): Reference column(s) to aggregate on in ref_df.
    :return: DataFrame: The aggregated dataframe.
    """

    df = df.copy()
    ref_df = ref_df.copy()

    ##Load model if string
    if isinstance(model, str):
        if openai_key is None and gemini_key is None and not _is_gemini_embedding_model(model):
            model = load_model(model)

    ## Just use the merge function with merge type 1:m
    df_lm_matched = merge(df, ref_df, merge_type="1:1", on=None, model=model, left_on=left_on,
                                right_on=right_on, suffixes=("_x", "_y"), batch_size=128,
                                 openai_key=openai_key, gemini_key=gemini_key)

    return df_lm_matched



def evaluate_pairs(df: DataFrame,
    model: Union[str, LinkTransformer] = "all-MiniLM-L6-v2",
    left_on: Union[str, List[str]]=None,
    right_on: Union[str, List[str]]=None,
    openai_key: str = None,
    gemini_key: str = None
    ) -> DataFrame:
    """
    This function evaluates paired columns in a dataframe and gives a match score (cosine similarity). 
    Typically, this can be though of as a way to evaluate already merged in dataframes.

    :param df (DataFrame): Dataframe to evaluate.
    :param model (str): Language model to use.
    :param left_on (Union[str, List[str]]): Column(s) to evaluate on in df.
    :param right_on (Union[str, List[str]]): Reference column(s) to evaluate on in df.
    :return: DataFrame: The evaluated dataframe.
    """

    df = df.copy()

    ##Load model if string
    if isinstance(model, str):
        if openai_key is None and gemini_key is None and not _is_gemini_embedding_model(model):
            model = load_model(model)

    ###We will serialize the columns if they are lists
    if isinstance(left_on, list):
        strings_left = serialize_columns(df, left_on, model=model)
    else:
        strings_left = df[left_on].tolist()
    
    if isinstance(right_on, list):
        strings_right = serialize_columns(df, right_on, model=model)
    else:
        strings_right = df[right_on].tolist()



    ## Infer embeddings for df1
    embeddings1 = infer_embeddings(strings_left, model, batch_size=128, openai_key=openai_key, gemini_key=gemini_key)
    ## Infer embeddings for df2
    embeddings2 = infer_embeddings(strings_right, model, batch_size=128, openai_key=openai_key, gemini_key=gemini_key)

    ### Expand dim if embeddings are 1d (numpy)
    if len(embeddings1.shape) == 1:
        embeddings1 = np.expand_dims(embeddings1, axis=0)
    if len(embeddings2.shape) == 1:
        embeddings2 = np.expand_dims(embeddings2, axis=0)
    ## Normalize embedding tensors using numpy

    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    ## Compute cosine similarity between CORRESPONDING pair of embeddings
    cosine_similarity_12 = cosine_similarity_corresponding_pairs(embeddings1,embeddings2)

    ## Add cosine similarity to df
    df["score"] = cosine_similarity_12.flatten()

    return df

def cluster_rows(
    df: DataFrame,
    model: Union[str, LinkTransformer] = "all-MiniLM-L6-v2",
    on: Union[str, List[str]]=None,
    cluster_type: str = "SLINK",
    cluster_params: Dict[str, Any] = {'threshold': 0.5, "min cluster size": 2, "metric": "cosine"},
    openai_key: str = None,
    gemini_key: str = None
) -> DataFrame:
    """
    Deduplicate a dataframe based on a similarity threshold. Various clustering options are supported.

    :param df (DataFrame): Dataframe to deduplicate.
    :param model (str): Language model to use.
    :param on (Union[str, List[str]]): Column(s) to cluster on.
    :param cluster_type (str): Clustering method to use. Defaults to "SLINK".
    :param cluster_params (Dict[str, Any]): Parameters for clustering method. Defaults to {'threshold': 0.5, "min cluster size": 2, "metric": "cosine"}.
    :param openai_key (str): OpenAI API key.

    Supported clustering methods and their parameters:
    
    - "agglomerative": 
        - "threshold": 0.5
        - "clustering linkage": "ward"
        - "metric": "euclidean"

    - "HDBScan": 
        - "min cluster size": 5
        - "min samples": 1
        - "metric": "cosine"

    - "SLINK": 
        - "min cluster size": 2
        - "threshold": 0.1
        - "metric": "cosine"

    :return: DataFrame: The deduplicated dataframe.
    """

    df = df.copy()

    ##Load model if string
    if isinstance(model, str):
        if openai_key is None and gemini_key is None and not _is_gemini_embedding_model(model):
            model = load_model(model)

    ## First, get the embeddings
    ### If len(on)>1, then we need to serialize the columns
    if isinstance(on, list):
        strings = serialize_columns(df, on, model=model)
    else:
        strings = df[on].tolist()
    
    ## Infer embeddings for df
    embeddings = infer_embeddings(strings, model, batch_size=128, openai_key=openai_key, gemini_key=gemini_key)
    ## Normalize embedding tensors using numpy
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    ### Now, cluster the embeddings based on similarity threshold
    labels = cluster(cluster_type, cluster_params, embeddings, corpus_ids=None)
    ### Now, keep only 1 row per cluster
    df["cluster"] = labels
    return df





def dedup_rows(
    df: DataFrame,
    model: Union[str, LinkTransformer] = "all-MiniLM-L6-v2",
    on: Union[str, List[str]]=None,
    cluster_type: str = "SLINK",
    cluster_params: Dict[str, Any] = {'threshold': 0.5, "min cluster size": 2, "metric": "cosine"},
    openai_key: str = None,
    gemini_key: str = None
) -> DataFrame:
    """
    Deduplicate a dataframe based on a similarity threshold. This is just clustering and keeping the first row in each cluster.
    Refer to the docs for the cluster_rows function for more details.

    :param df (DataFrame): Dataframe to deduplicate.
    :param model (str): Language model to use.
    :param on (Union[str, List[str]]): Column(s) to deduplicate on.
    :param cluster_type (str): Clustering method to use. Defaults to "SLINK".
    :param cluster_params (Dict[str, Any]): Parameters for clustering method. Defaults to {'threshold': 0.5, "min cluster size": 2, "metric": "cosine"}.
    :param openai_key (str): OpenAI API key
    :return: DataFrame: The deduplicated dataframe.
    """

    df = df.copy()

    ##Load model if string
    if isinstance(model, str):
        if openai_key is None and gemini_key is None and not _is_gemini_embedding_model(model):
            model = load_model(model)
    

    print(f"Deduplicating dataframe with originally {len(df)} rows")
    ##Drop exact duplicates
    print("Checking for and dropping exact duplicates")
    df = df.drop_duplicates(subset=on, keep="first")
    print(f"Number of rows after dropping exact duplicates: {len(df)}")

    df = cluster_rows(df, model, on, cluster_type, cluster_params, openai_key, gemini_key)
    df = df.drop_duplicates(subset="cluster", keep="first")
    df = df.drop(columns=["cluster"])
    print(f"Number of rows after deduplication: {len(df)}")

    return df


    

def all_pair_combos_evaluate(df: DataFrame,
    model: Union[str, LinkTransformer] = "all-MiniLM-L6-v2",
    left_on: Union[str, List[str]]=None,
    right_on: Union[str, List[str]]=None,
    openai_key: str = None,
    gemini_key: str = None
    ) -> DataFrame:
    """
    Get similarity scores for every pair of rows in a dataframe. 
    We make this efficient by only embedding each string once and get all possible pairwise distances
    and add the expanded rows and their scores to the dataframe
    :param df (DataFrame): Dataframe to evaluate.
    :param model (str): Language model to use.
    :param left_on (Union[str, List[str]]): Column(s) to evaluate on in df.
    :param right_on (Union[str, List[str]]): Reference column(s) to evaluate on in df.
    :param openai_key (str): OpenAI API key
    :return: DataFrame: The evaluated dataframe.
    """

    df = df.copy()

    ##Load model if string
    if isinstance(model, str):
        if openai_key is None and gemini_key is None and not _is_gemini_embedding_model(model):
            model = load_model(model)

    ###Get the embeddings for the left_on column
    if isinstance(left_on, list):
        strings_left = serialize_columns(df, left_on, model=model)
    else:
        strings_left = df[left_on].tolist()
    
    ## Infer embeddings for df1
    embeddings1 = infer_embeddings(strings_left, model, batch_size=128, openai_key=openai_key, gemini_key=gemini_key)
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)

    ###Get the embeddings for the right_on column
    if isinstance(right_on, list):
        strings_right = serialize_columns(df, right_on, model=model)
    else:
        strings_right = df[right_on].tolist()
    
    ## Infer embeddings for df1
    embeddings2 = infer_embeddings(strings_right, model, batch_size=128, openai_key=openai_key, gemini_key=gemini_key)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    ###calculate the cosine similarity between all pairs of rows
    cosine_similarity_12_all_pairs = cosine_similarity(embeddings1,embeddings2)
    ##Flatten the matrix
    cosine_similarity_12_all_pairs = cosine_similarity_12_all_pairs.flatten()

    ###This gives an n*n matrix of cosine similarities. 

    ###Similarly, make an n*n matrix of the left_on column and right_on column
    left_values = df[left_on].to_numpy()
    right_values = df[right_on].to_numpy()
    left_on_all_pairs = np.repeat(left_values, len(right_values), axis=0)


    ###Left part of the df was repeating n times. We also want the right one to repeat, but 


    right_on_all_pairs = np.tile(right_values, (len(left_values), 1))

    ###Now, flattenn the right on pairs
    right_on_all_pairs = right_on_all_pairs.flatten()

    ###Now, we can make a dataframe with the left_on, right_on and cosine similarity
    df_concat = pd.DataFrame({"left_on":left_on_all_pairs,"right_on":right_on_all_pairs,"score":cosine_similarity_12_all_pairs})

    return df_concat



def merge_knn(
    df1: DataFrame,
    df2: DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    model: Union[str, LinkTransformer] = "all-MiniLM-L6-v2",
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    k: int = 1,
    suffixes: Tuple[str, str] = ("_x", "_y"),
    batch_size: int = 128,
    openai_key: Optional[str] = None,
    gemini_key: Optional[str] = None,
    drop_sim_threshold: float = None,
) -> DataFrame:
    """
    Merge two dataframes using language model embeddings. This function would support k nearest neighbors matching for each row in df1.
    Merge is a special case of this function when k=1.
    :param df1 (DataFrame): First dataframe (left).
    :param df2 (DataFrame): Second dataframe (right).
    :param on (Union[str, List[str]], optional): Column(s) to join on in df1. Defaults to None.
    :param model (str): Language model to use.
    :param left_on (Union[str, List[str]], optional): Column(s) to join on in df1. Defaults to None.
    :param right_on (Union[str, List[str]], optional): Column(s) to join on in df2. Defaults to None.
    :param k (int): Number of nearest neighbors to match for each row in df1. Defaults to 1.
    :param suffixes (Tuple[str, str]): Suffixes to use for overlapping columns. Defaults to ('_x', '_y').
    :param batch_size (int): Batch size for inferencing embeddings. Defaults to 128.
    :param openai_key (str, optional): OpenAI API key for OpenAI models. Defaults to None.
    :param gemini_key (str, optional): Gemini API key for Gemini embedding models. Defaults to None.
    :param drop_sim_threshold (float, optional): Drop rows with similarity below this threshold. Defaults to None.
    :return: DataFrame: The merged dataframe.
    """


    ## Set common columns as on if not specified
    if on is None:
        on = list(set(df1.columns).intersection(set(df2.columns)))

    ## If left_on or right_on is not specified, set it to on
    if left_on is None:
        left_on = on
    if right_on is None:
        right_on = on

    on = None


    df1 = df1.copy()
    df2 = df2.copy()
    ## give ids to each df
    ##Ensure that there is no id_lt column in df1 or df2
    if "id_lt" in df1.columns:
        raise ValueError(f"Column id_lt already exists in df1, please rename it to proceed")
    if "id_lt" in df2.columns:
        raise ValueError(f"Column id_lt already exists in df2,please rename it to proceed")

    df1.loc[:, "id_lt"] = np.arange(len(df1))
    df2.loc[:, "id_lt"] = np.arange(len(df2))

    model_is_api_embedding = isinstance(model, str) and (
        _is_openai_embedding_model(model) or _is_gemini_embedding_model(model)
    )

    if isinstance(right_on, list):
        if model_is_api_embedding:
            strings_right = serialize_columns(df2, right_on, sep_token="<SEP>")
        else:
            strings_right = serialize_columns(df2, right_on, model=model)
    if isinstance(left_on, list):
        if model_is_api_embedding:
            strings_left = serialize_columns(df1, left_on, sep_token="<SEP>")
        else:
            strings_left = serialize_columns(df1, left_on, model=model)
    else:
        strings_left = df1[left_on].tolist()
        strings_right = df2[right_on].tolist()
    
    ## Load the model
    if isinstance(model, str):
        if openai_key is None and gemini_key is None and not _is_gemini_embedding_model(model):
            model = load_model(model)


    ## Infer embeddings for df1
    embeddings1 = infer_embeddings(strings_left, model, batch_size=batch_size, openai_key=openai_key, gemini_key=gemini_key, return_numpy= True)
    ## Infer embeddings for df2
    embeddings2 = infer_embeddings(strings_right, model, batch_size=batch_size, openai_key=openai_key, gemini_key=gemini_key, return_numpy= True)


    ### Expand dim if embeddings are 1d (numpy)
    if len(embeddings1.shape) == 1:
        embeddings1 = np.expand_dims(embeddings1, axis=0)
    if len(embeddings2.shape) == 1:
        embeddings2 = np.expand_dims(embeddings2, axis=0)

        

    ## Normalize embedding tensors using numpy
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)



    ## Create index
    index = faiss.IndexFlatIP(embeddings1.shape[1])
    
    print("Adding embeddings to index")


    index.add(embeddings2)

    print("Searching index")

    D, I = index.search(embeddings1, k)


    ## Check nearest neighbor of the first text in df1 as a test
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    ## Fuzzily merge the dfs based on the faiss index queries
    ###Each I sublst is a list of k nearest neighbors for each row in df1 in terms of indices of df2
    ###We need to expand the rows of df1 and df2 to match the number of rows in df1
    ###We also need to expand the scores to match the number of rows in df1

    ###First, expand the rows of df1
    df1_expanded = df1.loc[np.repeat(df1.index.values, k)].reset_index(drop=True)
    ###Now, expand the rows of df2
    df2_expanded = df2.iloc[I.flatten()].reset_index(drop=True)

    ###Now, merge the expanded dfs
    df_lm_matched = df1_expanded.merge(df2_expanded, left_index=True, right_index=True, how="inner",suffixes=suffixes)

    ### Add score column
    df_lm_matched["score"] =  D.flatten()

        
        

    if drop_sim_threshold is not None:
        df_lm_matched = df_lm_matched[df_lm_matched["score"]>=drop_sim_threshold]
        print(f"Dropped rows with similarity below {drop_sim_threshold}")

    print(f"LM matched on key columns - left: {left_on}{suffixes[0]}, right: {right_on}{suffixes[1]}")
        

    return df_lm_matched


def _coerce_llm_match_and_confidence(response_text: str) -> Tuple[int, float]:
    """
    Parse an LLM response into a binary match flag and confidence in [0, 1].
    Expected format is JSON with keys like `is_match` and `confidence`, but the
    parser is intentionally permissive for robustness.
    """
    is_match = 0
    confidence = 0.0

    try:
        payload = json.loads(response_text)
        if isinstance(payload, dict):
            raw_match = payload.get("is_match", payload.get("match", payload.get("label", 0)))
            if isinstance(raw_match, bool):
                is_match = int(raw_match)
            elif isinstance(raw_match, (int, float)):
                is_match = int(raw_match > 0)
            elif isinstance(raw_match, str):
                is_match = int(raw_match.strip().lower() in {"yes", "true", "1", "match"})

            raw_conf = payload.get("confidence", payload.get("score", payload.get("probability", 0.0)))
            try:
                confidence = float(raw_conf)
            except Exception:
                confidence = 0.0
        elif isinstance(payload, list) and len(payload) >= 2:
            try:
                is_match = int(float(payload[0]) > 0)
                confidence = float(payload[1])
            except Exception:
                pass
    except Exception:
        lower_txt = response_text.lower()
        if any(tok in lower_txt for tok in ["yes", "true", "match"]):
            is_match = 1

        score_matches = re.findall(r"([01](?:\.\d+)?)", response_text)
        if score_matches:
            try:
                confidence = float(score_matches[-1])
            except Exception:
                confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))
    return is_match, confidence


def _is_openai_embedding_model(model: Any) -> bool:
    if not isinstance(model, str):
        return False
    model_name = model.lower()
    return "text-embedding" in model_name or "ada-002" in model_name


def _resolve_knn_api_model(
    knn_api_model: Optional[str],
    openai_key: Optional[str],
    gemini_key: Optional[str],
) -> str:
    if knn_api_model is not None and knn_api_model != "auto":
        return knn_api_model

    if gemini_key or os.getenv("GEMINI_API_KEY"):
        return "gemini-embedding-001"
    if openai_key or os.getenv("OPENAI_API_KEY"):
        return "text-embedding-3-small"

    raise ValueError(
        "Could not resolve `knn_api_model`: provide `openai_key`/OPENAI_API_KEY or "
        "`gemini_key`/GEMINI_API_KEY, or pass an explicit `knn_api_model`."
    )


def _resolve_knn_retrieval_config(
    model: Union[str, LinkTransformer],
    knn_sbert_model: Optional[Union[str, LinkTransformer]],
    knn_api_model: Optional[str],
    openai_key: Optional[str],
    gemini_key: Optional[str],
) -> Tuple[Union[str, LinkTransformer], Optional[str], Optional[str]]:
    if knn_sbert_model is not None and knn_api_model is not None:
        raise ValueError("Specify only one of `knn_sbert_model` or `knn_api_model`, not both.")

    if knn_sbert_model is not None:
        return knn_sbert_model, None, None

    if knn_api_model is not None:
        resolved_api_model = _resolve_knn_api_model(knn_api_model, openai_key=openai_key, gemini_key=gemini_key)
        retrieval_openai_key = openai_key if _is_openai_embedding_model(resolved_api_model) else None
        retrieval_gemini_key = gemini_key if _is_gemini_embedding_model(resolved_api_model) else None
        return resolved_api_model, retrieval_openai_key, retrieval_gemini_key

    warnings.warn(
        "Using `model` as shared default. Key-resolved model behavior may apply for both KNN retrieval and classification. "
        "Specify `knn_sbert_model` for SBERT retrieval.",
        UserWarning,
    )

    retrieval_openai_key = openai_key if _is_openai_embedding_model(model) else None
    retrieval_gemini_key = gemini_key if _is_gemini_embedding_model(model) else None
    return model, retrieval_openai_key, retrieval_gemini_key


def _infer_retrieval_mode(model: Union[str, LinkTransformer]) -> str:
    if isinstance(model, str):
        if _is_gemini_embedding_model(model):
            return "api-gemini-embedding"
        if _is_openai_embedding_model(model):
            return "api-openai-embedding"
        return "sbert-or-local"
    return "loaded-linktransformer"


def merge_k_judge(
    df1: DataFrame,
    df2: DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    model: Union[str, LinkTransformer] = "all-MiniLM-L6-v2",
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    k: int = 3,
    suffixes: Tuple[str, str] = ("_x", "_y"),
    batch_size: int = 128,
    openai_key: Optional[str] = None,
    gemini_key: Optional[str] = None,
    knn_sbert_model: Optional[Union[str, LinkTransformer]] = None,
    knn_api_model: Optional[str] = None,
    drop_sim_threshold: float = None,
    judge_llm_model: Optional[str] = None,
    llm_provider: str = "auto",
    llm_prompt: Optional[str] = None,
    llm_params: Optional[Dict[str, Any]] = None,
    confidence_threshold: Optional[float] = None,
    max_retries: int = 5,
    ratelimit_sleep_time: int = 15,
) -> DataFrame:
    """
    Retrieve candidate matches with `merge_knn`, then verify each pair with an LLM.
    Adds:
    - `llm_is_match` (0/1)
    - `llm_confidence` (float in [0, 1])
    - `llm_raw_response` (raw model output)

    :param confidence_threshold: optional post-filter threshold on llm_confidence.
    """
    llm_params = llm_params or {}

    if llm_provider not in {"auto", "openai", "gemini"}:
        raise ValueError("llm_provider must be one of {'auto', 'openai', 'gemini'}")

    if judge_llm_model is None or (isinstance(judge_llm_model, str) and judge_llm_model.strip() == ""):
        raise ValueError(
            "merge_k_judge requires `judge_llm_model`. "
            "Use `merge_knn` if you do not want LLM-based judgement."
        )

    resolved_judge_model = judge_llm_model

    resolved_provider = llm_provider
    if resolved_provider == "auto":
        resolved_provider = "gemini" if "gemini" in str(resolved_judge_model).lower() else "openai"

    openai_api_key = openai_key or os.getenv("OPENAI_API_KEY")
    gemini_api_key = gemini_key or os.getenv("GEMINI_API_KEY")

    if resolved_provider == "openai" and openai_api_key is None:
        raise ValueError("merge_k_judge with llm_provider='openai' requires `openai_key` or OPENAI_API_KEY.")
    if resolved_provider == "gemini" and gemini_api_key is None:
        raise ValueError("merge_k_judge with llm_provider='gemini' requires `gemini_key` or GEMINI_API_KEY.")

    retrieval_model, retrieval_openai_key, retrieval_gemini_key = _resolve_knn_retrieval_config(
        model=model,
        knn_sbert_model=knn_sbert_model,
        knn_api_model=knn_api_model,
        openai_key=openai_key,
        gemini_key=gemini_key,
    )

    if knn_sbert_model is not None:
        retrieval_source = "knn_sbert_model"
    elif knn_api_model is not None:
        retrieval_source = "knn_api_model"
    else:
        retrieval_source = "model"

    judge_source = "judge_llm_model"
    resolution_msg = (
        "merge_k_judge resolution -> "
        f"retrieval_source={retrieval_source}, retrieval_model={retrieval_model}, "
        f"retrieval_mode={_infer_retrieval_mode(retrieval_model)}, "
        f"judge_source={judge_source}, judge_model={resolved_judge_model}, "
        f"judge_provider={resolved_provider}."
    )
    warnings.warn(resolution_msg, UserWarning)

    candidates = merge_knn(
        df1=df1,
        df2=df2,
        on=on,
        model=retrieval_model,
        left_on=left_on,
        right_on=right_on,
        k=k,
        suffixes=suffixes,
        batch_size=batch_size,
        openai_key=retrieval_openai_key,
        gemini_key=retrieval_gemini_key,
        drop_sim_threshold=drop_sim_threshold,
    ).copy()

    def _raise_judge_error(judge_error: Exception, stage: str) -> None:
        raise RuntimeError(
            "merge_k_judge failed during "
            f"{stage} (provider={resolved_provider}, model={resolved_judge_model}). "
            f"Underlying error: {repr(judge_error)}. "
            "Use merge_knn if you do not want LLM-based judgement."
        ) from judge_error

    if llm_prompt is None:
        llm_prompt = (
            "You are a fuzzy entity/text-matching judge. This could be an entity or just text descriptions that need matching. Compare LEFT and RIGHT records and decide if they refer to "
            "the same real-world entity. <SEP> signifies a concat of two variables in the record. Return ONLY compact JSON with keys: "
            "is_match (0 or 1) and confidence (float between 0 and 1)."
        )

    if isinstance(left_on, str):
        left_cols = [left_on]
    elif isinstance(left_on, list):
        left_cols = left_on
    elif isinstance(on, str):
        left_cols = [on]
    elif isinstance(on, list):
        left_cols = on
    else:
        left_cols = [c for c in df1.columns if c in df2.columns]

    if isinstance(right_on, str):
        right_cols = [right_on]
    elif isinstance(right_on, list):
        right_cols = right_on
    elif isinstance(on, str):
        right_cols = [on]
    elif isinstance(on, list):
        right_cols = on
    else:
        right_cols = left_cols

    openai_client = None
    gemini_model_client = None
    try:
        if resolved_provider == "openai":
            openai_client = openai.OpenAI(
                api_key=openai_api_key,
                timeout=llm_params["request_timeout"] if "request_timeout" in llm_params else 15,
            )
        else:
            try:
                import google.generativeai as genai
            except ImportError as exc:
                raise ImportError(
                    "Gemini LLM classification requires `google-generativeai`. Install it to use llm_provider='gemini'."
                ) from exc
            genai.configure(api_key=gemini_api_key)
            gemini_model_client = genai.GenerativeModel(model_name=resolved_judge_model)
    except Exception as judge_init_error:
        _raise_judge_error(judge_init_error, stage="judge client initialization")

    llm_matches: List[int] = []
    llm_confidences: List[float] = []
    llm_raw: List[str] = []

    for _, row in candidates.iterrows():
        left_payload = {}
        right_payload = {}

        for col in left_cols:
            col_name = f"{col}{suffixes[0]}" if f"{col}{suffixes[0]}" in candidates.columns else col
            left_payload[col] = None if col_name not in candidates.columns else row[col_name]

        for col in right_cols:
            col_name = f"{col}{suffixes[1]}" if f"{col}{suffixes[1]}" in candidates.columns else col
            right_payload[col] = None if col_name not in candidates.columns else row[col_name]

        user_content = json.dumps({"left": left_payload, "right": right_payload}, default=str)
        response_text = ""
        for retry in range(max_retries):
            try:
                if resolved_provider == "openai":
                    response = openai_client.chat.completions.create(
                        model=resolved_judge_model,
                        messages=[
                            {"role": "system", "content": llm_prompt},
                            {"role": "user", "content": user_content},
                        ],
                        temperature=llm_params["temperature"] if "temperature" in llm_params else 0,
                        max_tokens=llm_params["max_tokens"] if "max_tokens" in llm_params else 50,
                        top_p=llm_params["top_p"] if "top_p" in llm_params else 1,
                        frequency_penalty=llm_params["frequency_penalty"] if "frequency_penalty" in llm_params else 0,
                        presence_penalty=llm_params["presence_penalty"] if "presence_penalty" in llm_params else 0,
                    )
                    response_text = response.choices[0].message.content or ""
                else:
                    prompt_text = (
                        f"{llm_prompt}\n\n"
                        f"Compare the following pair payload and return JSON only.\n"
                        f"{user_content}"
                    )
                    response = gemini_model_client.generate_content(prompt_text)
                    response_text = getattr(response, "text", None) or ""
                break
            except Exception as judge_runtime_error:
                if retry == max_retries - 1:
                    _raise_judge_error(judge_runtime_error, stage="judge inference")
                else:
                    time.sleep(ratelimit_sleep_time * (2 ** retry))

        match_flag, match_conf = _coerce_llm_match_and_confidence(response_text)
        llm_matches.append(match_flag)
        llm_confidences.append(match_conf)
        llm_raw.append(response_text)

    candidates["llm_is_match"] = llm_matches
    candidates["llm_confidence"] = llm_confidences
    candidates["llm_raw_response"] = llm_raw

    if confidence_threshold is not None:
        candidates = candidates[candidates["llm_confidence"] >= confidence_threshold]

    return candidates


def merge_knn_with_llm(
    df1: DataFrame,
    df2: DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    model: Union[str, LinkTransformer] = "all-MiniLM-L6-v2",
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    k: int = 3,
    suffixes: Tuple[str, str] = ("_x", "_y"),
    batch_size: int = 128,
    openai_key: Optional[str] = None,
    gemini_key: Optional[str] = None,
    knn_sbert_model: Optional[Union[str, LinkTransformer]] = None,
    knn_api_model: Optional[str] = None,
    drop_sim_threshold: float = None,
    judge_llm_model: Optional[str] = None,
    llm_provider: str = "auto",
    llm_prompt: Optional[str] = None,
    llm_params: Optional[Dict[str, Any]] = None,
    confidence_threshold: Optional[float] = None,
    max_retries: int = 5,
    ratelimit_sleep_time: int = 15,
) -> DataFrame:
    """Backward-compatible alias for `merge_k_judge`."""
    return merge_k_judge(
        df1=df1,
        df2=df2,
        on=on,
        model=model,
        left_on=left_on,
        right_on=right_on,
        k=k,
        suffixes=suffixes,
        batch_size=batch_size,
        openai_key=openai_key,
        gemini_key=gemini_key,
        knn_sbert_model=knn_sbert_model,
        knn_api_model=knn_api_model,
        drop_sim_threshold=drop_sim_threshold,
        judge_llm_model=judge_llm_model,
        llm_provider=llm_provider,
        llm_prompt=llm_prompt,
        llm_params=llm_params,
        confidence_threshold=confidence_threshold,
        max_retries=max_retries,
        ratelimit_sleep_time=ratelimit_sleep_time,
    )


def merge_knn_openai(
    df1: DataFrame,
    df2: DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    model: Union[str, LinkTransformer] = "all-MiniLM-L6-v2",
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    k: int = 3,
    suffixes: Tuple[str, str] = ("_x", "_y"),
    batch_size: int = 128,
    openai_key: Optional[str] = None,
    gemini_key: Optional[str] = None,
    knn_sbert_model: Optional[Union[str, LinkTransformer]] = None,
    knn_api_model: Optional[str] = None,
    drop_sim_threshold: float = None,
    judge_llm_model: Optional[str] = None,
    llm_provider: str = "openai",
    llm_prompt: Optional[str] = None,
    llm_params: Optional[Dict[str, Any]] = None,
    confidence_threshold: Optional[float] = None,
    max_retries: int = 5,
    ratelimit_sleep_time: int = 15,
) -> DataFrame:
    """Backward-compatible alias for `merge_k_judge`."""
    return merge_k_judge(
        df1=df1,
        df2=df2,
        on=on,
        model=model,
        left_on=left_on,
        right_on=right_on,
        k=k,
        suffixes=suffixes,
        batch_size=batch_size,
        openai_key=openai_key,
        gemini_key=gemini_key,
        knn_sbert_model=knn_sbert_model,
        knn_api_model=knn_api_model,
        drop_sim_threshold=drop_sim_threshold,
        judge_llm_model=judge_llm_model,
        llm_provider=llm_provider,
        llm_prompt=llm_prompt,
        llm_params=llm_params,
        confidence_threshold=confidence_threshold,
        max_retries=max_retries,
        ratelimit_sleep_time=ratelimit_sleep_time,
    )


def classify_rows(
    df: DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    model: str = None,
    num_labels: int = 2,
    label_map: Optional[dict] = None,
    batch_size: int = 128,
    openai_key: Optional[str] = None,
    openai_topic: Optional[str] = None,
    openai_prompt: Optional[str] = None,
    openai_params: Optional[dict] = {},
    progress_bar: bool = True
):
    """
    Classify texts in all rows of one or more columns whether they are relevant to a certain topic. The function uses
    either a trained classifier to make predictions or an OpenAI API key to send requests and retrieve classification
    results from ChatCompletion endpoint. The function returns a copy of the input dataframe with a new column "clf_preds_{on}" that stores the
    classification results.

    :param df: (DataFrame) the dataframe.
    :param on: (Union[str, List[str]], optional) Column(s) to classify (if multiple columns are passed in, they will be joined).
    :param model: (str) filepath to the model to use (to use OpenAI, see "https://platform.openai.com/docs/models").
    :param num_labels: (int) number of labels to predict. Defaults to 2.
    :param label_map: (dict) a dictionary that maps text labels to numeric labels. Used for OpenAI predictions.
    :param batch_size: (int) Batch size for inferencing embeddings. Defaults to 128.
    :param openai_key: (str, optional) OpenAI API key for InferKit API. Defaults to None.
    :param openai_topic: (str, optional) The topic predict whether the text is relevant or not. Defaults to None.
    :param openai_prompt: (str, optional) Custom system prompt for OpenAI ChatCompletion endpoint. Defaults to None.
    :param openai_params: (str, optional) Custom parameters for OpenAI ChatCompletion endpoint. Defaults to None.
    :param progress_bar: (bool) Whether to show progress bar. Defaults to True.
    :returns: DataFrame: The dataframe with a new column "clf_preds_{on}" that stores the classification results.
    """

    df = df.copy()

    # load label dict from model path if exists
    if os.path.exists(os.path.join(model, "label_map.json")):
        with open(os.path.join(model, "label_map.json"), encoding="utf-8") as f:
            label_map = json.load(f)

    # check if label dict is compatible with num of labels
    if label_map is not None and len(label_map) != num_labels:
        raise ValueError(f"label_dict has {label_map} entries, but num_labels is {num_labels}. ")

    # check if the parameters for any inference method is properly specified
    if openai_key is None and model is None:
        raise ValueError("Must either specify a model or use an OpenAI key")

    # infer using HF or local models
    if openai_key is None:
        clf = load_clf(model, num_labels=num_labels)

        # join texts in columns if needed
        if isinstance(on, list):
            strings_col = serialize_columns(df, on, model=model)
        else:
            strings_col = df[on].tolist()

        # tokenize data
        tokenized_data = tokenize_data_for_inference(strings_col, "inf_data", model)
        disable_tqdm = not progress_bar
        # initialize trainer
        inference_args = TrainingArguments(output_dir="save", per_device_eval_batch_size=batch_size,disable_tqdm=disable_tqdm)
        trainer = Trainer(model=clf, args=inference_args)

        # predict and save results
        predictions = trainer.predict(tokenized_data)
        preds = np.argmax(predictions.predictions, axis=-1)

        # convert numeric labels to text labels according to label_map if exists
        if label_map is not None:
            reversed_label_map = {val: key for (key, val) in label_map.items()}
            try:
                preds = [reversed_label_map[pred] for pred in preds]
            except:
                warnings.warn("Failed to convert from numeric labels to text labels. Text labels are kept. ")

        assert len(preds) == df.shape[0], "DEBUG: Length mismatch"

        if isinstance(on, str):
            df[f"clf_preds_{on}"] = preds
        else:
            df[f"clf_preds_{'-'.join(on)}"] = preds

    # infer using OpenAI GPT API
    else:
        if openai_topic is None and openai_prompt is None:
            raise ValueError("Must provide either openai_topic or openai_prompt to use OpenAI classification")

        # use default label dict if it is not provided
        if label_map is None:
            label_map = {"Yes": 1, "No": 0}
            if openai_prompt is not None:
                warnings.warn("You are using a customized prompt but a default label map (Yes/No mapping). ")

        # check if label dict is compatible with num of labels
        if len(label_map) != num_labels:
            raise ValueError(f"label_dict has {label_map} entries, but num_labels is {num_labels}. ")

        # join texts in columns if needed
        if isinstance(on, list):
            strings_col = serialize_columns(df, on, sep_token=" ")
        else:
            strings_col = df[on].tolist()

        # get predictions from openai
        preds = predict_rows_with_openai(
            strings_col, model, openai_key, openai_topic, openai_prompt, openai_params, label_map
        )

        assert len(preds) == df.shape[0], "DEBUG: Length mismatch"

        if isinstance(on, str):
            df[f"clf_preds_{on}"] = preds
        else:
            df[f"clf_preds_{'-'.join(on)}"] = preds

    return df


def transform_rows(
    df: pd.DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    *,
    openai_key: Optional[str] = None,
    model: str = "gpt-4o",
    openai_prompt: Optional[str] = None,
    openai_params: Optional[Dict[str, Any]] = None,
    batch_size: int = 50,
    output_column: Optional[str] = None,
    progress_bar: bool = True
) -> pd.DataFrame:
    """
    High-level wrapper: transform every cell in one or more columns via any batch transform fn.
    By default uses OpenAI with a “Fix spelling mistakes” prompt.

    :param df:            (pd.DataFrame)
    :param on:            (str or list[str]) column(s) to transform
    :param openai_key:    (str) your OpenAI API key
    :param openai_model:  (str) which model to call
    :param openai_prompt: (str) system prompt; if None defaults to:
                           "Fix spelling mistakes in each of the following strings.
                            Return a JSON array of the corrected strings."
    :param openai_params: (dict) extra ChatCompletion params, plus:
                           - request_timeout
                           - max_retries
                           - ratelimit_sleep_time
    :param batch_size:    (int) number of rows in each chunk
    :param output_column: (str) name for the new column; default "transformed_{on}"
    :param progress_bar:  (bool)
    :returns:             pd.DataFrame with an added column of transforms
    """
    df = df.copy()
    openai_params = openai_params or {}

    if on is None:
        raise ValueError("Must specify `on=` to select which column(s) to transform.")
    if openai_key is None:
        raise ValueError("Must provide `openai_key` to use OpenAI.")

    # default prompt
    if openai_prompt is None:
        openai_prompt = (
            "Fix spelling mistakes in each of the following strings. "
        )
    
    print(f"Transforming column(s) {on} with OpenAI model {model} using prompt: {openai_prompt}")

    # serialize multi-column if needed
    if isinstance(on, list):
        tmp_col = "__lt_tmp__"
        df[tmp_col] = serialize_columns(df, on, sep_token=" ")
        column_to_use = tmp_col
        key_name = "-".join(on)
    else:
        column_to_use = on
        key_name = on
    
    ##drop if on is na in any row
    df = df.dropna(subset=[column_to_use])

    # final output column name
    out_col = output_column or f"transformed_{key_name}"

    # prepare the OpenAI client and fn_kwargs
    client = openai.OpenAI(
        api_key=openai_key,
        timeout=openai_params.get("request_timeout", 10)
    )
    fn_kwargs = {
        "client": client,
        "model": model,
        "prompt": openai_prompt,
        "max_retries": openai_params.get("max_retries", 5),
        "ratelimit_sleep_time": openai_params.get("ratelimit_sleep_time", 15),
        "openai_params": openai_params
    }

    # use the generic transform_column util
    result = transform_column(
        df=df,
        column=column_to_use,
        transform_fn=openai_transform,
        fn_kwargs=fn_kwargs,
        chunk_size=batch_size,
        output_column=out_col,
        progress_bar=progress_bar
    )

    # clean up temp if used
    if isinstance(on, list):
        result = result.drop(columns=[tmp_col])

    return result
