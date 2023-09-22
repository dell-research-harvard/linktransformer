###Inference and Linkage script
###We want to link dfs together using embeddings
import json
import os
import warnings

import numpy as np
import pandas as pd
import faiss
from typing import Union, List, Optional, Tuple,Dict, Any
from pandas import DataFrame

from linktransformer.modified_sbert.cluster_fns import cluster
# from linktransformer.utils import serialize_columns, infer_embeddings, load_model, load_clf, cosine_similarity_corresponding_pairs, tokenize_data_for_inference, predict_rows_with_openai
from linktransformer.utils import *
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from transformers import TrainingArguments, Trainer



def merge(
    df1: DataFrame,
    df2: DataFrame,
    merge_type: str = '1:1',
    on: Optional[Union[str, List[str]]] = None,
    model: str = "all-MiniLM-L6-v2",
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    suffixes: Tuple[str, str] = ('_x', '_y'),
    use_gpu: bool = False,
    batch_size: int = 128,
    openai_key: Optional[str] = None
) -> DataFrame:
    """
    Merge two dataframes using language model embeddings.

    :param df1 (DataFrame): First dataframe (left).
    :param df2 (DataFrame): Second dataframe (right).
    :param merge_type (str): Type of merge to perform (1:m or m:1 or 1:1).
    :param model (str): Language model to use.
    :param on (Union[str, List[str]], optional): Column(s) to join on in df1. Defaults to None.
    :param left_on (Union[str, List[str]], optional): Column(s) to join on in df1. Defaults to None.
    :param right_on (Union[str, List[str]], optional): Column(s) to join on in df2. Defaults to None.
    :param suffixes (Tuple[str, str]): Suffixes to use for overlapping columns. Defaults to ('_x', '_y').
    :param use_gpu (bool): Whether to use GPU. Not supported yet. Defaults to False.
    :param batch_size (int): Batch size for inferencing embeddings. Defaults to 128.
    :param openai_key (str, optional): OpenAI API key for InferKit API. Defaults to None.
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

    ### If how is 1:m, then we want to merge df1 to df2
    ### Check if how is valid
    if merge_type not in ["1:m", "m:1", "1:1"]:
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

    if isinstance(right_on, list):
        strings_right = serialize_columns(df2, right_on, model=model)
    if isinstance(left_on, list):
        strings_left = serialize_columns(df1, left_on, model=model)
    else:
        strings_left = df1[left_on].tolist()
        strings_right = df2[right_on].tolist()

    ## Load the model
    if openai_key is None:
        model = load_model(model)

    ## Infer embeddings for df1
    embeddings1 = infer_embeddings(strings_left, model, batch_size=batch_size, openai_key=openai_key)
    ## Infer embeddings for df2
    embeddings2 = infer_embeddings(strings_right, model, batch_size=batch_size, openai_key=openai_key)

    ### Expand dim if embeddings are 1d (numpy)
    if len(embeddings1.shape) == 1:
        embeddings1 = np.expand_dims(embeddings1, axis=0)
    if len(embeddings2.shape) == 1:
        embeddings2 = np.expand_dims(embeddings2, axis=0)
    ## Normalize embedding tensors using numpy

    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
    ## Create index
    if use_gpu:
        raise ValueError(f"GPU not supported yet")
    else:
        index = faiss.IndexFlatIP(embeddings1.shape[1])

    ## Add to index depending on merge type
    if merge_type == "1:m":
        index.add(embeddings2)
    elif merge_type == "m:1":
        index.add(embeddings2)
    elif merge_type == "1:1":
        index.add(embeddings2)

    ## Search index
    if merge_type == "1:m":
        D, I = index.search(embeddings1, 1)
    elif merge_type == "m:1":
        D, I = index.search(embeddings1, 1)
    elif merge_type == "1:1":
        D, I = index.search(embeddings1, 1)

    ## Check nearest neighbor of the first text in df1 as a test
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    ## Fuzzily merge the dfs based on the faiss index queries
    df_lm_matched = df1.merge(df2.iloc[I.flatten()].reset_index(drop=True), left_index=True, right_index=True, how="inner")
    ### Add score column
    df_lm_matched["score"] = D.flatten()

    print(f"LM matched on key columns - left: {left_on}{suffixes[0]}, right: {right_on}{suffixes[1]}")
    return df_lm_matched

    


def merge_blocking(
    df1: DataFrame,
    df2: DataFrame,
    merge_type: str = '1:1',
    on: Optional[Union[str, List[str]]] = None,
    model: str = "all-MiniLM-L6-v2",
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    blocking_vars: Optional[List[str]] = None,
    suffixes: Tuple[str, str] = ('_x', '_y'),
    use_gpu: bool = False,
    batch_size: int = 128,
    openai_key: Optional[str] = None
) -> DataFrame:
    """
    Merge two dataframes using language model embeddings with optional blocking.

    :param df1 (DataFrame): First dataframe (left).
    :param df2 (DataFrame): Second dataframe (right).
    :param merge_type (str): Type of merge to perform (1:m or m:1 or 1:1).
    :param model (str): Language model to use.
    :param on (Union[str, List[str]], optional): Column(s) to join on in df1. Defaults to None.
    :param left_on (Union[str, List[str]], optional): Column(s) to join on in df1. Defaults to None.
    :param right_on (Union[str, List[str]], optional): Column(s) to join on in df2. Defaults to None.
    :param blocking_vars (List[str], optional): Columns to use for blocking. Defaults to None.
    :param suffixes (Tuple[str, str]): Suffixes to use for overlapping columns. Defaults to ('_x', '_y').
    :param use_gpu (bool): Whether to use GPU. Not supported yet. Defaults to False.
    :param batch_size (int): Batch size for inferencing embeddings. Defaults to 128.
    :param openai_key (str, optional): OpenAI API key for InferKit API. Defaults to None.
    :return: DataFrame: The merged dataframe.
    """
    ### For blocking, we need to chunk the dfs into blocks
    ### First, we need to check if blocking vars are specified
    if blocking_vars is None:
        print("No blocking vars specified, matching between all rows")
        df_lm_matched = merge(df1, df2, merge_type=merge_type, on=on, model=model, left_on=left_on,
                                    right_on=right_on, suffixes=suffixes, use_gpu=use_gpu, batch_size=batch_size,
                                     openai_key=openai_key)
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
        num_blocks_1 = len(df1.groupby(blocking_vars))
        num_blocks_2 = len(df2.groupby(blocking_vars))
        print(f"Number of blocks in df1: {num_blocks_1}")
        print(f"Number of blocks in df2: {num_blocks_2}")

        ## Now, get the blocks
        df1_blocks = df1.groupby(blocking_vars)
        df2_blocks = df2.groupby(blocking_vars)

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

        for block_1 in common_keys:
            print(f"Merging block {block_1}")
            df1_block = df1_blocks.get_group(block_1)
            df2_block = df2_blocks.get_group(block_1)
            ## Merge the blocks
            df_block_matched = merge(df1_block, df2_block, merge_type=merge_type, on=on, model=model,
                                           left_on=left_on, right_on=right_on, suffixes=suffixes, use_gpu=use_gpu,
                                           batch_size=batch_size, openai_key=openai_key)
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
    model: str,
    left_on: Union[str, List[str]],
    right_on: Union[str, List[str]],
    openai_key: str = None
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

    ## Just use the merge function with merge type 1:m
    df_lm_matched = merge(df, ref_df, merge_type="1:1", on=None, model=model, left_on=left_on,
                                right_on=right_on, suffixes=("_x", "_y"), use_gpu=False, batch_size=128,
                                 openai_key=openai_key)

    return df_lm_matched



def evaluate_pairs(df,model,left_on,right_on,openai_key=None):
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

    ###We will serialize the columns if they are lists
    if isinstance(left_on, list):
        strings_left = serialize_columns(df, left_on, model=model)
    else:
        strings_left = df[left_on].tolist()
    
    if isinstance(right_on, list):
        strings_right = serialize_columns(df, right_on, model=model)
    else:
        strings_right = df[right_on].tolist()

    ## Load the model
    if openai_key is None:
        model = load_model(model)

    ## Infer embeddings for df1
    embeddings1 = infer_embeddings(strings_left, model, batch_size=128, openai_key=openai_key)
    ## Infer embeddings for df2
    embeddings2 = infer_embeddings(strings_right, model, batch_size=128, openai_key=openai_key)

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
    model: str,
    on: Union[str, List[str]],
    cluster_type: str = "SLINK",
    cluster_params: Dict[str, Any] = {'threshold': 0.5, "min cluster size": 2, "metric": "cosine"},
    openai_key: str = None
) -> DataFrame:
    """
    Deduplicate a dataframe based on a similarity threshold. Various clustering options are supported.         
    "agglomerative": {
            "threshold": 0.5,
            "clustering linkage": "ward",  # You can choose a default linkage method
            "metric": "euclidean",  # You can choose a default metric
        },
        "HDBScan": {
            "min cluster size": 5,
            "min samples": 1,
        },
        "SLINK": {
            "min cluster size": 2,
            "threshold": 0.1,
        },
    }

    :param df (DataFrame): Dataframe to deduplicate.
    :param model (str): Language model to use.
    :param on (Union[str, List[str]]): Column(s) to deduplicate on.
    :param cluster_type (str): Clustering method to use. Defaults to "SLINK".
    :param cluster_params (Dict[str, Any]): Parameters for clustering method. Defaults to {'threshold': 0.5, "min cluster size": 2, "metric": "cosine"}.
    :param openai_key (str): OpenAI API key
    :return: DataFrame: The deduplicated dataframe.
    """

    df = df.copy()


    ## First, get the embeddings
    ### If len(on)>1, then we need to serialize the columns
    if isinstance(on, list):
        strings = serialize_columns(df, on, model=model)
    else:
        strings = df[on].tolist()
    
    ## Infer embeddings for df
    embeddings = infer_embeddings(strings, model, batch_size=128, openai_key=openai_key)
    ## Normalize embedding tensors using numpy
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    ### Now, cluster the embeddings based on similarity threshold
    labels = cluster(cluster_type, cluster_params, embeddings, corpus_ids=None)
    ### Now, keep only 1 row per cluster
    df["cluster"] = labels
    return df





def dedup_rows(
    df: DataFrame,
    model: str,
    on: Union[str, List[str]],
    cluster_type: str = "SLINK",
    cluster_params: Dict[str, Any] = {'threshold': 0.5, "min cluster size": 2, "metric": "cosine"},
    openai_key: str = None
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

    

    print(f"Deduplicating dataframe with originally {len(df)} rows")
    ##Drop exact duplicates
    print("Checking for and dropping exact duplicates")
    df = df.drop_duplicates(subset=on, keep="first")
    print(f"Number of rows after dropping exact duplicates: {len(df)}")

    df = cluster_rows(df, model, on, cluster_type, cluster_params, openai_key)
    df = df.drop_duplicates(subset="cluster", keep="first")
    df = df.drop(columns=["cluster"])
    print(f"Number of rows after deduplication: {len(df)}")

    return df


    

def all_pair_combos_evaluate(df,model,left_on,right_on,openai_key=None):
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

    ###Get the embeddings for the left_on column
    if isinstance(left_on, list):
        strings_left = serialize_columns(df, left_on, model=model)
    else:
        strings_left = df[left_on].tolist()
    
    ## Infer embeddings for df1
    embeddings1 = infer_embeddings(strings_left, model, batch_size=128, openai_key=openai_key)
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)

    ###Get the embeddings for the right_on column
    if isinstance(right_on, list):
        strings_right = serialize_columns(df, right_on, model=model)
    else:
        strings_right = df[right_on].tolist()
    
    ## Infer embeddings for df1
    embeddings2 = infer_embeddings(strings_right, model, batch_size=128, openai_key=openai_key)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    ###calculate the cosine similarity between all pairs of rows
    cosine_similarity_12_all_pairs = cosine_similarity(embeddings1,embeddings2)
    ##Flatten the matrix
    cosine_similarity_12_all_pairs = cosine_similarity_12_all_pairs.flatten()

    ###This gives an n*n matrix of cosine similarities. 

    ###Similarly, make an n*n matrix of the left_on column and right_on column
    left_on_all_pairs = np.repeat(df[left_on].values, len(df[right_on]), axis=0)


    ###Left part of the df was repeating n times. We also want the right one to repeat, but 


    right_on_all_pairs = np.tile(df[right_on].values, (len(df[left_on]),1))

    ###Now, flattenn the right on pairs
    right_on_all_pairs = right_on_all_pairs.flatten()

    ###Now, we can make a dataframe with the left_on, right_on and cosine similarity
    df_concat = pd.DataFrame({"left_on":left_on_all_pairs,"right_on":right_on_all_pairs,"score":cosine_similarity_12_all_pairs})

    return df_concat



def merge_knn(
    df1: DataFrame,
    df2: DataFrame,
    merge_type: str = '1:1',
    on: Optional[Union[str, List[str]]] = None,
    model: str = "all-MiniLM-L6-v2",
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    k: int = 1,
    suffixes: Tuple[str, str] = ('_x', '_y'),
    use_gpu: bool = False,
    batch_size: int = 128,
    openai_key: Optional[str] = None,
    drop_sim_threshold: float = None
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
    :param use_gpu (bool): Whether to use GPU. Not supported yet. Defaults to False.
    :param batch_size (int): Batch size for inferencing embeddings. Defaults to 128.
    :param openai_key (str, optional): OpenAI API key for InferKit API. Defaults to None.
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

    ### If how is 1:m, then we want to merge df1 to df2
    ### Check if how is valid
    if merge_type not in ["1:m", "m:1", "1:1"]:
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

    if isinstance(right_on, list):
        strings_right = serialize_columns(df2, right_on, model=model)
    if isinstance(left_on, list):
        strings_left = serialize_columns(df1, left_on, model=model)
    else:
        strings_left = df1[left_on].tolist()
        strings_right = df2[right_on].tolist()
    
    ## Load the model
    model = load_model(model)

    ## Infer embeddings for df1
    embeddings1 = infer_embeddings(strings_left, model, batch_size=batch_size, openai_key=openai_key)
    ## Infer embeddings for df2
    embeddings2 = infer_embeddings(strings_right, model, batch_size=batch_size, openai_key=openai_key)

    ### Expand dim if embeddings are 1d (numpy)
    if len(embeddings1.shape) == 1:
        embeddings1 = np.expand_dims(embeddings1, axis=0)
    if len(embeddings2.shape) == 1:
        embeddings2 = np.expand_dims(embeddings2, axis=0)

    ## Normalize embedding tensors using numpy

    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    ## Create index
    if use_gpu:
        raise ValueError(f"GPU not supported yet")
    else:
        index = faiss.IndexFlatIP(embeddings1.shape[1])

    ## Add to index depending on merge type
    if merge_type == "1:m":
        index.add(embeddings2)
    elif merge_type == "m:1":
        index.add(embeddings2)
    elif merge_type == "1:1":
        index.add(embeddings2)

    ## Search index
    if merge_type == "1:m":
        D, I = index.search(embeddings1, k)
    elif merge_type == "m:1":
        D, I = index.search(embeddings1, k)
    elif merge_type == "1:1":
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
    df_lm_matched = df1_expanded.merge(df2_expanded, left_index=True, right_index=True, how="inner")

    ### Add score column
    df_lm_matched["score"] =  D.flatten()

    print(f"LM matched on key columns - left: {left_on}{suffixes[0]}, right: {right_on}{suffixes[1]}")

    if drop_sim_threshold is not None:
        df_lm_matched = df_lm_matched[df_lm_matched["score"]>=drop_sim_threshold]
        print(f"Dropped rows with similarity below {drop_sim_threshold}")

    return df_lm_matched


def classify_rows(
    df: DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    model: str = None,
    num_labels: int = 2,
    label_map: Optional[dict] = None,
    use_gpu: bool = False,
    batch_size: int = 128,
    openai_key: Optional[str] = None,
    openai_topic: Optional[str] = None,
    openai_prompt: Optional[str] = None,
    openai_params: Optional[dict] = {}
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
    :param use_gpu: (bool) Whether to use GPU. Not supported yet. Defaults to False.
    :param batch_size: (int) Batch size for inferencing embeddings. Defaults to 128.
    :param openai_key: (str, optional) OpenAI API key for InferKit API. Defaults to None.
    :param openai_topic: (str, optional) The topic predict whether the text is relevant or not. Defaults to None.
    :param openai_prompt: (str, optional) Custom system prompt for OpenAI ChatCompletion endpoint. Defaults to None.
    :param openai_params: (str, optional) Custom parameters for OpenAI ChatCompletion endpoint. Defaults to None.
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

        # initialize trainer
        inference_args = TrainingArguments(output_dir="save", per_device_eval_batch_size=batch_size)
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













##Add a function to test nn of each row within a df. 
