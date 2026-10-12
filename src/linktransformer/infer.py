###Inference and Linkage script
###We want to link dfs together using embeddings
import json
import logging
import os
import re
import time
import warnings
import openai
import threading
import concurrent.futures

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Tuple,Dict, Any
from pandas import DataFrame

from linktransformer.cluster_fns import cluster
from linktransformer.modelling.LinkTransformer import LinkTransformer
from linktransformer.utils import (
    _gemini_client,
    _gemini_generate_text,
    _is_gemini_embedding_model,
    _is_openai_embedding_model,
    _load_tokenizer,
    _openai_chat_completion_resilient,
    gemini_transform,
    infer_embeddings,
    load_clf,
    load_model,
    openai_transform,
    predict_rows_with_llm,
    serialize_columns,
    transform_column,
    cosine_similarity_corresponding_pairs,
)
from linktransformer.search import knn_search, range_search
from tqdm import tqdm
from linktransformer.api_runtime import (
    append_jsonl,
    log_failure,
    parse_retry_policy,
    read_jsonl,
    call_with_retry,
)


logger = logging.getLogger(__name__)

DEFAULT_CLUSTER_PARAMS = {'threshold': 0.5, "min cluster size": 2, "metric": "cosine"}


def _shared_columns_in_left_order(df1: DataFrame, df2: DataFrame) -> List[str]:
    return [col for col in df1.columns if col in df2.columns]


def _build_blocking_unmatched_frame(
    block_df: DataFrame,
    *,
    side: str,
    df1_cols: List[str],
    df2_cols: List[str],
    suffixes: Tuple[str, str],
) -> DataFrame:
    if side not in {"left", "right"}:
        raise ValueError("side must be either 'left' or 'right'")

    left_renamed_cols = [f"{col}{suffixes[0]}" if col in df2_cols else col for col in df1_cols]
    right_renamed_cols = [f"{col}{suffixes[1]}" if col in df1_cols else col for col in df2_cols]

    if side == "left":
        left_map = {orig: renamed for orig, renamed in zip(df1_cols, left_renamed_cols)}
        out = block_df.rename(columns=left_map).copy()
        for col in right_renamed_cols:
            out[col] = np.nan
        out["match_status"] = "left_only_block"
    else:
        right_map = {orig: renamed for orig, renamed in zip(df2_cols, right_renamed_cols)}
        out = block_df.rename(columns=right_map).copy()
        for col in left_renamed_cols:
            out[col] = np.nan
        out["match_status"] = "right_only_block"

    out["score"] = np.nan

    ordered_cols = left_renamed_cols + right_renamed_cols + ["score", "match_status"]
    out = out.reindex(columns=ordered_cols)
    return out


def _path_with_tag(path: Optional[str], tag: str) -> Optional[str]:
    if path is None:
        return None
    if path.endswith(".jsonl"):
        return path[:-6] + f"_{tag}.jsonl"
    return path + f"_{tag}.jsonl"



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
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    failure_log_path: Optional[str] = None,
    retry_policy: Optional[Dict[str, Any]] = None,
    embedding_cache: Optional[str] = None,
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
    :param embedding_cache (str, optional): Folder for an on-disk embedding cache. Embeddings are stored per model and text, so repeated calls only embed texts that are not cached yet. Defaults to None.
    :return: DataFrame: The merged dataframe.
    """
    ## Set common columns as on if not specified
    if on is None:
        on = _shared_columns_in_left_order(df1, df2)

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
           warnings.warn("Keys in df1 are not unique")

    if merge_type == "m:1":
        ## Check if keys in df2 are unique
        if df2[right_on].duplicated().any():
            warnings.warn("Keys in df2 are not unique")
    if merge_type == "1:1":
        ## Check if keys in df1 are unique
        if df1[left_on].duplicated().any():
           warnings.warn("Keys in df1 are not unique")
        ## Check if keys in df2 are unique
        if df2[right_on].duplicated().any():
           warnings.warn("Keys in df2 are not unique")

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
        if openai_key is None and gemini_key is None and not _is_gemini_embedding_model(model) and not _is_openai_embedding_model(model):
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
    embeddings1 = infer_embeddings(
        strings_left,
        model,
        batch_size=batch_size,
        openai_key=openai_key,
        gemini_key=gemini_key,
        checkpoint_path=_path_with_tag(checkpoint_path, "merge_left_embeddings"),
        resume=resume,
        failure_log_path=failure_log_path,
        retry_policy=retry_policy,
        embedding_cache=embedding_cache,
    )
    ## Infer embeddings for df2
    embeddings2 = infer_embeddings(
        strings_right,
        model,
        batch_size=batch_size,
        openai_key=openai_key,
        gemini_key=gemini_key,
        checkpoint_path=_path_with_tag(checkpoint_path, "merge_right_embeddings"),
        resume=resume,
        failure_log_path=failure_log_path,
        retry_policy=retry_policy,
        embedding_cache=embedding_cache,
    )

    ### Expand dim if embeddings are 1d (numpy)
    if len(embeddings1.shape) == 1:
        embeddings1 = np.expand_dims(embeddings1, axis=0)
    if len(embeddings2.shape) == 1:
        embeddings2 = np.expand_dims(embeddings2, axis=0)
    ## Normalize embedding tensors using numpy

    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    ## Nearest neighbour in df2 for each row of df1
    D, I = knn_search(embeddings1, embeddings2, 1)

    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    ## Fuzzily merge the dfs based on the nearest-neighbour search
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
    embedding_cache: Optional[str] = None,
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
    :param embedding_cache (str, optional): Folder for an on-disk embedding cache. Embeddings are stored per model and text, so repeated calls only embed texts that are not cached yet. Defaults to None.
    :return: DataFrame: The merged dataframe.
    """
    ### For blocking, we need to chunk the dfs into blocks
    ### First, we need to check if blocking vars are specified
    if blocking_vars is None:
        logger.info("No blocking vars specified, matching between all rows")
        df_lm_matched = merge(df1, df2, merge_type=merge_type, on=on, model=model, left_on=left_on,
                                    right_on=right_on, suffixes=suffixes, batch_size=batch_size,
                                     openai_key=openai_key, gemini_key=gemini_key, embedding_cache=embedding_cache)
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
        logger.info(f"Number of blocks in df1: {num_blocks_1}")
        logger.info(f"Number of blocks in df2: {num_blocks_2}")

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
            if openai_key is None and gemini_key is None and not _is_gemini_embedding_model(model) and not _is_openai_embedding_model(model):
                model = load_model(model)      

        for block_1 in df1_block_list:
            if block_1 not in common_keys:
                continue
            logger.info(f"Merging block {block_1}")
            df1_block = df1_blocks.get_group(block_1)
            df2_block = df2_blocks.get_group(block_1)
            ## Merge the blocks
            df_block_matched = merge(df1_block, df2_block, merge_type=merge_type, on=on, model=model,
                                           left_on=left_on, right_on=right_on, suffixes=suffixes,
                                           batch_size=batch_size, openai_key=openai_key, gemini_key=gemini_key, embedding_cache=embedding_cache)
            df_block_matched["match_status"] = "matched_block"
            ## Add to merged dfs
            merged_dfs.append(df_block_matched)

        ## Add df corresponding to skipped keys
        for block_1 in skipped_keys_1:
            df1_block = df1_blocks.get_group(block_1)
            merged_dfs.append(
                _build_blocking_unmatched_frame(
                    df1_block,
                    side="left",
                    df1_cols=df1.columns.tolist(),
                    df2_cols=df2.columns.tolist(),
                    suffixes=suffixes,
                )
            )

        for block_2 in skipped_keys_2:
            df2_block = df2_blocks.get_group(block_2)
            merged_dfs.append(
                _build_blocking_unmatched_frame(
                    df2_block,
                    side="right",
                    df1_cols=df1.columns.tolist(),
                    df2_cols=df2.columns.tolist(),
                    suffixes=suffixes,
                )
            )

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
    embedding_cache: Optional[str] = None,
) -> DataFrame:
    """
    Aggregate the dataframe based on a reference dataframe using a language model.

    :param df (DataFrame): Dataframe to aggregate.
    :param ref_df (DataFrame): Reference dataframe to aggregate on.
    :param model (str): Language model to use.
    :param left_on (Union[str, List[str]]): Column(s) to aggregate on in df.
    :param right_on (Union[str, List[str]]): Reference column(s) to aggregate on in ref_df.
    :param embedding_cache (str, optional): Folder for an on-disk embedding cache. Embeddings are stored per model and text, so repeated calls only embed texts that are not cached yet. Defaults to None.
    :return: DataFrame: The aggregated dataframe.
    """

    df = df.copy()
    ref_df = ref_df.copy()

    ##Load model if string
    if isinstance(model, str):
        if openai_key is None and gemini_key is None and not _is_gemini_embedding_model(model) and not _is_openai_embedding_model(model):
            model = load_model(model)

    ## Each row of df is matched to its nearest row of ref_df
    df_lm_matched = merge(df, ref_df, on=None, model=model, left_on=left_on,
                                right_on=right_on, suffixes=("_x", "_y"), batch_size=128,
                                 openai_key=openai_key, gemini_key=gemini_key, embedding_cache=embedding_cache)

    return df_lm_matched



def evaluate_pairs(df: DataFrame,
    model: Union[str, LinkTransformer] = "all-MiniLM-L6-v2",
    left_on: Union[str, List[str]]=None,
    right_on: Union[str, List[str]]=None,
    openai_key: str = None,
    gemini_key: str = None,
    embedding_cache: Optional[str] = None,
    ) -> DataFrame:
    """
    This function evaluates paired columns in a dataframe and gives a match score (cosine similarity). 
    Typically, this can be though of as a way to evaluate already merged in dataframes.

    :param df (DataFrame): Dataframe to evaluate.
    :param model (str): Language model to use.
    :param left_on (Union[str, List[str]]): Column(s) to evaluate on in df.
    :param right_on (Union[str, List[str]]): Reference column(s) to evaluate on in df.
    :param embedding_cache (str, optional): Folder for an on-disk embedding cache. Embeddings are stored per model and text, so repeated calls only embed texts that are not cached yet. Defaults to None.
    :return: DataFrame: The evaluated dataframe.
    """

    df = df.copy()

    ##Load model if string
    if isinstance(model, str):
        if openai_key is None and gemini_key is None and not _is_gemini_embedding_model(model) and not _is_openai_embedding_model(model):
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
    embeddings1 = infer_embeddings(strings_left, model, batch_size=128, openai_key=openai_key, gemini_key=gemini_key, embedding_cache=embedding_cache)
    ## Infer embeddings for df2
    embeddings2 = infer_embeddings(strings_right, model, batch_size=128, openai_key=openai_key, gemini_key=gemini_key, embedding_cache=embedding_cache)

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
    cluster_params: Optional[Dict[str, Any]] = None,
    openai_key: str = None,
    gemini_key: str = None,
    embedding_cache: Optional[str] = None,
) -> DataFrame:
    """
    Deduplicate a dataframe based on a similarity threshold. Various clustering options are supported.

    :param df (DataFrame): Dataframe to deduplicate.
    :param model (str): Language model to use.
    :param on (Union[str, List[str]]): Column(s) to cluster on.
    :param cluster_type (str): Clustering method to use. Defaults to "SLINK".
    :param cluster_params (Dict[str, Any], optional): Parameters for clustering method. Defaults to DEFAULT_CLUSTER_PARAMS ({'threshold': 0.5, "min cluster size": 2, "metric": "cosine"}).
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

    :param embedding_cache (str, optional): Folder for an on-disk embedding cache. Embeddings are stored per model and text, so repeated calls only embed texts that are not cached yet. Defaults to None.

    :return: DataFrame: The deduplicated dataframe.
    """

    df = df.copy()
    cluster_params = dict(DEFAULT_CLUSTER_PARAMS if cluster_params is None else cluster_params)

    ##Load model if string
    if isinstance(model, str):
        if openai_key is None and gemini_key is None and not _is_gemini_embedding_model(model) and not _is_openai_embedding_model(model):
            model = load_model(model)

    ## First, get the embeddings
    ### If len(on)>1, then we need to serialize the columns
    if isinstance(on, list):
        strings = serialize_columns(df, on, model=model)
    else:
        strings = df[on].tolist()
    
    ## Infer embeddings for df
    embeddings = infer_embeddings(strings, model, batch_size=128, openai_key=openai_key, gemini_key=gemini_key, embedding_cache=embedding_cache)
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
    cluster_params: Optional[Dict[str, Any]] = None,
    openai_key: str = None,
    gemini_key: str = None,
    embedding_cache: Optional[str] = None,
) -> DataFrame:
    """
    Deduplicate a dataframe based on a similarity threshold. This is just clustering and keeping the first row in each cluster.
    Refer to the docs for the cluster_rows function for more details.

    :param df (DataFrame): Dataframe to deduplicate.
    :param model (str): Language model to use.
    :param on (Union[str, List[str]]): Column(s) to deduplicate on.
    :param cluster_type (str): Clustering method to use. Defaults to "SLINK".
    :param cluster_params (Dict[str, Any], optional): Parameters for clustering method. Defaults to DEFAULT_CLUSTER_PARAMS ({'threshold': 0.5, "min cluster size": 2, "metric": "cosine"}).
    :param openai_key (str): OpenAI API key
    :param embedding_cache (str, optional): Folder for an on-disk embedding cache. Embeddings are stored per model and text, so repeated calls only embed texts that are not cached yet. Defaults to None.
    :return: DataFrame: The deduplicated dataframe.
    """

    df = df.copy()

    ##Load model if string
    if isinstance(model, str):
        if openai_key is None and gemini_key is None and not _is_gemini_embedding_model(model) and not _is_openai_embedding_model(model):
            model = load_model(model)
    

    logger.info(f"Deduplicating dataframe with originally {len(df)} rows")
    ##Drop exact duplicates
    logger.info("Checking for and dropping exact duplicates")
    df = df.drop_duplicates(subset=on, keep="first")
    logger.info(f"Number of rows after dropping exact duplicates: {len(df)}")

    df = cluster_rows(df, model, on, cluster_type, cluster_params, openai_key, gemini_key, embedding_cache=embedding_cache)
    df = df.drop_duplicates(subset="cluster", keep="first")
    df = df.drop(columns=["cluster"])
    logger.info(f"Number of rows after deduplication: {len(df)}")

    return df


    

def all_pair_combos_evaluate(df: DataFrame,
    model: Union[str, LinkTransformer] = "all-MiniLM-L6-v2",
    left_on: Union[str, List[str]]=None,
    right_on: Union[str, List[str]]=None,
    openai_key: str = None,
    gemini_key: str = None,
    embedding_cache: Optional[str] = None,
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
    :param embedding_cache (str, optional): Folder for an on-disk embedding cache. Embeddings are stored per model and text, so repeated calls only embed texts that are not cached yet. Defaults to None.
    :return: DataFrame: The evaluated dataframe.
    """

    df = df.copy()

    ##Load model if string
    if isinstance(model, str):
        if openai_key is None and gemini_key is None and not _is_gemini_embedding_model(model) and not _is_openai_embedding_model(model):
            model = load_model(model)

    ###Get the embeddings for the left_on column
    if isinstance(left_on, list):
        strings_left = serialize_columns(df, left_on, model=model)
    else:
        strings_left = df[left_on].tolist()
    
    ## Infer embeddings for df1
    embeddings1 = infer_embeddings(strings_left, model, batch_size=128, openai_key=openai_key, gemini_key=gemini_key, embedding_cache=embedding_cache)
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)

    ###Get the embeddings for the right_on column
    if isinstance(right_on, list):
        strings_right = serialize_columns(df, right_on, model=model)
    else:
        strings_right = df[right_on].tolist()
    
    ## Infer embeddings for df1
    embeddings2 = infer_embeddings(strings_right, model, batch_size=128, openai_key=openai_key, gemini_key=gemini_key, embedding_cache=embedding_cache)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    ###calculate the cosine similarity between all pairs of rows
    cosine_similarity_12_all_pairs = embeddings1 @ embeddings2.T  # embeddings are already normalized
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
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    failure_log_path: Optional[str] = None,
    retry_policy: Optional[Dict[str, Any]] = None,
    embedding_cache: Optional[str] = None,
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
    :param embedding_cache (str, optional): Folder for an on-disk embedding cache. Embeddings are stored per model and text, so repeated calls only embed texts that are not cached yet. Defaults to None.
    :return: DataFrame: The merged dataframe.
    """


    ## Set common columns as on if not specified
    if on is None:
        on = _shared_columns_in_left_order(df1, df2)

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
    else:
        strings_right = df2[right_on].tolist()

    if isinstance(left_on, list):
        if model_is_api_embedding:
            strings_left = serialize_columns(df1, left_on, sep_token="<SEP>")
        else:
            strings_left = serialize_columns(df1, left_on, model=model)
    else:
        strings_left = df1[left_on].tolist()
    
    ## Load the model
    if isinstance(model, str):
        if openai_key is None and gemini_key is None and not _is_gemini_embedding_model(model) and not _is_openai_embedding_model(model):
            model = load_model(model)


    ## Infer embeddings for df1
    embeddings1 = infer_embeddings(
        strings_left,
        model,
        batch_size=batch_size,
        openai_key=openai_key,
        gemini_key=gemini_key,
        return_numpy=True,
        checkpoint_path=_path_with_tag(checkpoint_path, "merge_knn_left_embeddings"),
        resume=resume,
        failure_log_path=failure_log_path,
        retry_policy=retry_policy,
        embedding_cache=embedding_cache,
    )
    ## Infer embeddings for df2
    embeddings2 = infer_embeddings(
        strings_right,
        model,
        batch_size=batch_size,
        openai_key=openai_key,
        gemini_key=gemini_key,
        return_numpy=True,
        checkpoint_path=_path_with_tag(checkpoint_path, "merge_knn_right_embeddings"),
        resume=resume,
        failure_log_path=failure_log_path,
        retry_policy=retry_policy,
        embedding_cache=embedding_cache,
    )


    ### Expand dim if embeddings are 1d (numpy)
    if len(embeddings1.shape) == 1:
        embeddings1 = np.expand_dims(embeddings1, axis=0)
    if len(embeddings2.shape) == 1:
        embeddings2 = np.expand_dims(embeddings2, axis=0)

        

    ## Normalize embedding tensors using numpy
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)



    ## k nearest neighbours in df2 for each row of df1 (k is capped at len(df2))
    D, I = knn_search(embeddings1, embeddings2, k)
    k = I.shape[1]

    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    ## Fuzzily merge the dfs based on the nearest-neighbour search
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
        logger.info(f"Dropped rows with similarity below {drop_sim_threshold}")

    logger.info(f"LM matched on key columns - left: {left_on}{suffixes[0]}, right: {right_on}{suffixes[1]}")
        

    return df_lm_matched


def merge_range(
    df1: DataFrame,
    df2: DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    model: Union[str, LinkTransformer] = "all-MiniLM-L6-v2",
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    sim_threshold: float = 0.5,
    suffixes: Tuple[str, str] = ("_x", "_y"),
    batch_size: int = 128,
    openai_key: Optional[str] = None,
    gemini_key: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    failure_log_path: Optional[str] = None,
    retry_policy: Optional[Dict[str, Any]] = None,
    embedding_cache: Optional[str] = None,
) -> DataFrame:
    """
    Range-based left merge (exact range search over normalized embeddings).

    For each row in ``df1``, returns all rows in ``df2`` with cosine similarity
    greater than or equal to ``sim_threshold``. If a left row has no matches
    above threshold, it is retained with null right-side values and null score.

    :param df1 (DataFrame): First dataframe (left).
    :param df2 (DataFrame): Second dataframe (right).
    :param on (Union[str, List[str]], optional): Column(s) to join on in both dataframes.
    :param model (str or LinkTransformer): Embedding model to use.
    :param left_on (Union[str, List[str]], optional): Left key column(s). Defaults to ``on``.
    :param right_on (Union[str, List[str]], optional): Right key column(s). Defaults to ``on``.
    :param sim_threshold (float): Cosine similarity threshold for keeping matches.
    :param suffixes (Tuple[str, str]): Suffixes for overlapping columns.
    :param batch_size (int): Batch size for embedding inference.
    :param openai_key (str, optional): OpenAI API key for OpenAI embedding models.
    :param gemini_key (str, optional): Gemini API key for Gemini embedding models.
    :param embedding_cache (str, optional): Folder for an on-disk embedding cache. Embeddings are stored per model and text, so repeated calls only embed texts that are not cached yet. Defaults to None.
    :return: DataFrame: Left-join style matched dataframe with ``score`` column.
    """
    if sim_threshold < -1.0 or sim_threshold > 1.0:
        raise ValueError("sim_threshold must be between -1.0 and 1.0 for cosine similarity.")

    if on is None:
        on = _shared_columns_in_left_order(df1, df2)

    if left_on is None:
        left_on = on
    if right_on is None:
        right_on = on

    df1 = df1.copy()
    df2 = df2.copy()

    if "id_lt" in df1.columns:
        raise ValueError("Column id_lt already exists in df1, please rename it to proceed")
    if "id_lt" in df2.columns:
        raise ValueError("Column id_lt already exists in df2,please rename it to proceed")

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
    else:
        strings_right = df2[right_on].tolist()

    if isinstance(left_on, list):
        if model_is_api_embedding:
            strings_left = serialize_columns(df1, left_on, sep_token="<SEP>")
        else:
            strings_left = serialize_columns(df1, left_on, model=model)
    else:
        strings_left = df1[left_on].tolist()

    if isinstance(model, str):
        if openai_key is None and gemini_key is None and not _is_gemini_embedding_model(model) and not _is_openai_embedding_model(model):
            model = load_model(model)

    embeddings1 = infer_embeddings(
        strings_left,
        model,
        batch_size=batch_size,
        openai_key=openai_key,
        gemini_key=gemini_key,
        return_numpy=True,
        checkpoint_path=_path_with_tag(checkpoint_path, "merge_range_left_embeddings"),
        resume=resume,
        failure_log_path=failure_log_path,
        retry_policy=retry_policy,
        embedding_cache=embedding_cache,
    )
    embeddings2 = infer_embeddings(
        strings_right,
        model,
        batch_size=batch_size,
        openai_key=openai_key,
        gemini_key=gemini_key,
        return_numpy=True,
        checkpoint_path=_path_with_tag(checkpoint_path, "merge_range_right_embeddings"),
        resume=resume,
        failure_log_path=failure_log_path,
        retry_policy=retry_policy,
        embedding_cache=embedding_cache,
    )

    if len(embeddings1.shape) == 1:
        embeddings1 = np.expand_dims(embeddings1, axis=0)
    if len(embeddings2.shape) == 1:
        embeddings2 = np.expand_dims(embeddings2, axis=0)

    embeddings1 = embeddings1 / np.clip(np.linalg.norm(embeddings1, axis=1, keepdims=True), 1e-12, None)
    embeddings2 = embeddings2 / np.clip(np.linalg.norm(embeddings2, axis=1, keepdims=True), 1e-12, None)

    lims, distances, indices = range_search(embeddings1, embeddings2, sim_threshold)

    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    per_left_counts = np.diff(lims)
    total_matches = int(per_left_counts.sum())

    matched_frames = []

    if total_matches > 0:
        left_indices = np.repeat(np.arange(len(df1), dtype=np.int64), per_left_counts.astype(np.int64))
        right_indices = indices.astype(np.int64)

        df1_expanded = df1.iloc[left_indices].reset_index(drop=True)
        df2_expanded = df2.iloc[right_indices].reset_index(drop=True)

        matched_df = df1_expanded.merge(
            df2_expanded,
            left_index=True,
            right_index=True,
            how="inner",
            suffixes=suffixes,
        )
        matched_df["score"] = distances
        matched_df["__left_order"] = left_indices
        matched_frames.append(matched_df)

    unmatched_left_indices = np.where(per_left_counts == 0)[0]
    if len(unmatched_left_indices) > 0:
        unmatched_left = df1.iloc[unmatched_left_indices].reset_index(drop=True)
        unmatched_df = unmatched_left.merge(
            df2.iloc[[]],
            left_index=True,
            right_index=True,
            how="left",
            suffixes=suffixes,
        )
        unmatched_df["score"] = np.nan
        unmatched_df["__left_order"] = unmatched_left_indices
        matched_frames.append(unmatched_df)

    df_lm_matched = pd.concat(matched_frames, ignore_index=True)
    df_lm_matched = df_lm_matched.sort_values("__left_order", kind="stable").drop(columns=["__left_order"]).reset_index(drop=True)

    logger.info(
        f"Range matched rows: {len(df_lm_matched)} (threshold={sim_threshold}) on key columns "
        f"left={left_on}{suffixes[0]}, right={right_on}{suffixes[1]}"
    )

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


def _normalize_columns_arg(cols: Optional[Union[str, List[str]]], *, arg_name: str) -> Optional[List[str]]:
    if cols is None:
        return None
    if isinstance(cols, str):
        return [cols]
    if isinstance(cols, list) and all(isinstance(c, str) for c in cols):
        return cols
    raise TypeError(f"{arg_name} must be a string, list[str], or None")


def _resolve_judge_columns(
    *,
    df1: DataFrame,
    df2: DataFrame,
    on: Optional[Union[str, List[str]]],
    left_on: Optional[Union[str, List[str]]],
    right_on: Optional[Union[str, List[str]]],
    judge_on: Optional[Union[str, List[str]]],
    judge_left_on: Optional[Union[str, List[str]]],
    judge_right_on: Optional[Union[str, List[str]]],
) -> Tuple[List[str], List[str]]:
    """Resolve which columns are serialized into the LLM judging payload.

    If judge_* args are not provided, this preserves the existing behavior:
    - prefer left_on/right_on if set
    - else use on
    - else fall back to shared columns
    """

    # First, honor explicit judge_* args.
    judge_left_cols = _normalize_columns_arg(judge_left_on, arg_name="judge_left_on")
    judge_right_cols = _normalize_columns_arg(judge_right_on, arg_name="judge_right_on")
    judge_on_cols = _normalize_columns_arg(judge_on, arg_name="judge_on")

    if judge_on_cols is not None:
        if judge_left_cols is None:
            judge_left_cols = judge_on_cols
        if judge_right_cols is None:
            judge_right_cols = judge_on_cols

    # If only one side was provided explicitly, assume symmetric names.
    if judge_left_cols is not None and judge_right_cols is None:
        judge_right_cols = judge_left_cols
    if judge_right_cols is not None and judge_left_cols is None:
        judge_left_cols = judge_right_cols

    if judge_left_cols is not None and judge_right_cols is not None:
        return judge_left_cols, judge_right_cols

    # Backward-compatible fallback behavior.
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

    return left_cols, right_cols


def _is_missing_judgement_value(value: Any) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _row_has_complete_judgement(row: pd.Series) -> bool:
    required_cols = ["llm_is_match", "llm_confidence", "llm_raw_response"]
    for col in required_cols:
        if col not in row.index:
            return False
        if _is_missing_judgement_value(row[col]):
            return False
    return True


def judge_merge_k_out(
    candidates_df: DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    suffixes: Tuple[str, str] = ("_x", "_y"),
    openai_key: Optional[str] = None,
    gemini_key: Optional[str] = None,
    judge_llm_model: Optional[str] = None,
    llm_provider: str = "auto",
    llm_prompt: Optional[str] = None,
    llm_params: Optional[Dict[str, Any]] = None,
    max_retries: int = 5,
    ratelimit_sleep_time: int = 15,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    failure_log_path: Optional[str] = None,
    retry_policy: Optional[Dict[str, Any]] = None,
    judge_on: Optional[Union[str, List[str]]] = None,
    judge_left_on: Optional[Union[str, List[str]]] = None,
    judge_right_on: Optional[Union[str, List[str]]] = None,
    judge_skip_lower_thresh: Optional[float] = None,
    judge_skip_upper_thresh: Optional[float] = None,
    confidence_threshold: Optional[float] = None,
    progress_bar: bool = True,
    judge_workers: int = 1,
    failure_function_name: str = "judge_merge_k_out",
) -> DataFrame:
    """Judge an existing merge_k/merge_knn candidate output dataframe with an LLM.

    This utility supports resume via JSONL checkpoint, skipping already-judged rows,
    score-based short-circuiting, and multi-worker API-call concurrency.

    Adds/updates debugging columns:
    - `llm_input_payload`: serialized JSON payload sent to the judge LLM
    - `llm_is_match`, `llm_confidence`, `llm_raw_response`
    """
    llm_params = llm_params or {}
    candidates = candidates_df.copy()

    logger.info(
        "[judge_merge_k_out] Loaded candidate set "
        f"with {len(candidates)} rows."
    )

    if judge_workers < 1:
        raise ValueError("judge_workers must be >= 1")

    if llm_provider not in {"auto", "openai", "gemini"}:
        raise ValueError("llm_provider must be one of {'auto', 'openai', 'gemini'}")

    if judge_llm_model is None or (isinstance(judge_llm_model, str) and judge_llm_model.strip() == ""):
        raise ValueError("judge_merge_k_out requires `judge_llm_model`.")

    resolved_judge_model = judge_llm_model
    resolved_provider = llm_provider
    if resolved_provider == "auto":
        resolved_provider = "gemini" if "gemini" in str(resolved_judge_model).lower() else "openai"

    logger.info(
        "[judge_merge_k_out] Judge configuration -> "
        f"provider={resolved_provider}, model={resolved_judge_model}, "
        f"workers={judge_workers}."
    )

    openai_api_key = openai_key or os.getenv("OPENAI_API_KEY")
    gemini_api_key = gemini_key or os.getenv("GEMINI_API_KEY")
    if resolved_provider == "openai" and openai_api_key is None:
        raise ValueError("judge_merge_k_out with llm_provider='openai' requires `openai_key` or OPENAI_API_KEY.")
    if resolved_provider == "gemini" and gemini_api_key is None:
        raise ValueError("judge_merge_k_out with llm_provider='gemini' requires `gemini_key` or GEMINI_API_KEY.")

    if llm_prompt is None:
        llm_prompt = (
            "You are a fuzzy entity/text-matching judge. This could be an entity or just text descriptions that need matching. Compare LEFT and RIGHT records and decide if they refer to "
            "the same real-world entity. <SEP> signifies a concat of two variables in the record. Return ONLY compact JSON with keys: "
            "is_match (0 or 1) and confidence (float between 0 and 1)."
        )

    left_cols, right_cols = _resolve_judge_columns(
        df1=pd.DataFrame(columns=[]),
        df2=pd.DataFrame(columns=[]),
        on=on,
        left_on=left_on,
        right_on=right_on,
        judge_on=judge_on,
        judge_left_on=judge_left_on,
        judge_right_on=judge_right_on,
    )

    if len(left_cols) == 0 and len(right_cols) == 0:
        inferred_cols: List[str] = []
        for col in candidates.columns:
            if col.endswith(suffixes[0]):
                base = col[: -len(suffixes[0])]
                if f"{base}{suffixes[1]}" in candidates.columns:
                    inferred_cols.append(base)
        left_cols = inferred_cols
        right_cols = inferred_cols

    if len(left_cols) == 0 and len(right_cols) == 0:
        raise ValueError(
            "Could not infer judge columns from candidates_df. "
            "Provide `judge_on` or `judge_left_on`/`judge_right_on`."
        )

    for col in ["llm_input_payload", "llm_is_match", "llm_confidence", "llm_raw_response"]:
        if col not in candidates.columns:
            candidates[col] = np.nan

    llm_inputs = candidates["llm_input_payload"].tolist()
    llm_matches = candidates["llm_is_match"].tolist()
    llm_confidences = candidates["llm_confidence"].tolist()
    llm_raw = candidates["llm_raw_response"].tolist()

    policy = parse_retry_policy(
        retry_policy,
        default_max_retries=max_retries,
        default_base_sleep=float(ratelimit_sleep_time),
    )

    resumed_rows: Dict[int, Dict[str, Any]] = {}
    if checkpoint_path is not None and resume:
        logger.info(f"[judge_merge_k_out] Reading checkpoint: {checkpoint_path}")
        for rec in read_jsonl(checkpoint_path):
            if rec.get("kind") == "merge_k_judge_row" and "index" in rec:
                resumed_rows[int(rec["index"])] = rec
        logger.info(
            "[judge_merge_k_out] Loaded "
            f"{len(resumed_rows)} prior judged rows from checkpoint."
        )
    elif checkpoint_path is not None:
        logger.info(
            "[judge_merge_k_out] Checkpoint path provided but resume=False; "
            "new progress will be appended."
        )

    def _append_checkpoint(
        index: int,
        input_payload: str,
        match_flag: int,
        match_conf: float,
        response_text: str,
    ) -> None:
        append_jsonl(
            checkpoint_path,
            {
                "kind": "merge_k_judge_row",
                "index": index,
                "llm_input_payload": input_payload,
                "llm_is_match": match_flag,
                "llm_confidence": match_conf,
                "llm_raw_response": response_text,
            },
        )

    def _set_row_result(
        index: int,
        match_flag: int,
        match_conf: float,
        response_text: str,
        *,
        input_payload: Optional[str] = None,
        write_checkpoint: bool = True,
    ) -> None:
        if input_payload is not None:
            llm_inputs[index] = input_payload
        llm_matches[index] = match_flag
        llm_confidences[index] = match_conf
        llm_raw[index] = response_text
        if write_checkpoint:
            checkpoint_input = llm_inputs[index]
            if _is_missing_judgement_value(checkpoint_input):
                checkpoint_input = ""
            _append_checkpoint(index, str(checkpoint_input), match_flag, match_conf, response_text)

    def _raise_judge_error(judge_error: Exception, stage: str, item_key: Optional[int] = None) -> None:
        if item_key is not None:
            log_failure(
                failure_log_path,
                function_name=failure_function_name,
                provider=resolved_provider,
                model=resolved_judge_model,
                item_key=item_key,
                attempt=policy.max_retries,
                error=judge_error,
            )
        raise RuntimeError(
            "judge_merge_k_out failed during "
            f"{stage} (provider={resolved_provider}, model={resolved_judge_model}). "
            f"Underlying error: {repr(judge_error)}. "
            "Use merge_knn if you do not want LLM-based judgement."
        ) from judge_error

    thread_local = threading.local()

    def _get_openai_client() -> Any:
        client = getattr(thread_local, "openai_client", None)
        if client is None:
            client = openai.OpenAI(
                api_key=openai_api_key,
                timeout=llm_params.get("request_timeout", 15),
            )
            thread_local.openai_client = client
        return client

    def _get_gemini_client() -> Any:
        client = getattr(thread_local, "gemini_client", None)
        if client is None:
            client = _gemini_client(gemini_api_key, llm_params.get("request_timeout"))
            thread_local.gemini_client = client
        return client

    def _judge_once(user_content: str) -> str:
        def _judge_call() -> str:
            if resolved_provider == "openai":
                client = _get_openai_client()
                request_kwargs: Dict[str, Any] = {
                    "model": resolved_judge_model,
                    "messages": [
                        {"role": "system", "content": llm_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": llm_params.get("temperature", 0),
                    "max_tokens": llm_params.get("max_tokens", 50),
                    "top_p": llm_params.get("top_p", 1),
                    "frequency_penalty": llm_params.get("frequency_penalty", 0),
                    "presence_penalty": llm_params.get("presence_penalty", 0),
                }

                response = _openai_chat_completion_resilient(client, request_kwargs)
                return response.choices[0].message.content or ""
            gemini_client = _get_gemini_client()
            prompt_text = (
                f"{llm_prompt}\n\n"
                f"Compare the following pair payload and return JSON only.\n"
                f"{user_content}"
            )

            generation_config: Dict[str, Any] = {}
            if "temperature" in llm_params:
                generation_config["temperature"] = llm_params.get("temperature")
            if "top_p" in llm_params:
                generation_config["top_p"] = llm_params.get("top_p")
            if "top_k" in llm_params:
                generation_config["top_k"] = llm_params.get("top_k")
            if "max_output_tokens" in llm_params:
                generation_config["max_output_tokens"] = llm_params.get("max_output_tokens")
            elif "max_tokens" in llm_params:
                generation_config["max_output_tokens"] = llm_params.get("max_tokens")

            dropped_params: List[str] = []
            while True:
                try:
                    response_text = _gemini_generate_text(
                        gemini_client, resolved_judge_model, prompt_text, generation_config
                    )
                    break
                except Exception as exc:
                    error_text = str(exc)
                    unsupported_param: Optional[str] = None

                    match = re.search(
                        r"unsupported parameter(?:\s*[:=])?\s*['\"]?([a-zA-Z_][a-zA-Z0-9_]*)['\"]?",
                        error_text,
                        flags=re.IGNORECASE,
                    )
                    if match:
                        unsupported_param = match.group(1)
                    else:
                        match = re.search(
                            r"parameter\s+['\"]?([a-zA-Z_][a-zA-Z0-9_]*)['\"]?.*not supported",
                            error_text,
                            flags=re.IGNORECASE,
                        )
                        if match:
                            unsupported_param = match.group(1)

                    if unsupported_param and unsupported_param in generation_config:
                        generation_config.pop(unsupported_param, None)
                        dropped_params.append(unsupported_param)
                        continue

                    raise

            if dropped_params:
                warnings.warn(
                    "Gemini judge call dropped unsupported parameters for model "
                    f"{resolved_judge_model}: {sorted(set(dropped_params))}",
                    UserWarning,
                )
            return response_text

        return call_with_retry(_judge_call, policy)

    pending_items: List[Tuple[int, str]] = []
    existing_judged_count = 0
    resumed_applied_count = 0
    skipped_by_score_count = 0

    progress = tqdm(total=len(candidates), disable=not progress_bar, desc="judging matches")

    for idx, (_, row) in enumerate(candidates.iterrows()):
        if _row_has_complete_judgement(row):
            existing_judged_count += 1
            progress.update(1)
            continue

        if idx in resumed_rows:
            rec = resumed_rows[idx]
            _set_row_result(
                idx,
                int(rec.get("llm_is_match", 0)),
                float(rec.get("llm_confidence", 0.0)),
                str(rec.get("llm_raw_response", "")),
                input_payload=str(rec.get("llm_input_payload", "")),
                write_checkpoint=False,
            )
            resumed_applied_count += 1
            progress.update(1)
            continue

        skip_by_score = False
        if "score" in candidates.columns and pd.notna(row["score"]):
            row_score = float(row["score"])
            if judge_skip_lower_thresh is not None and row_score <= judge_skip_lower_thresh:
                skip_by_score = True
            if judge_skip_upper_thresh is not None and row_score >= judge_skip_upper_thresh:
                skip_by_score = True

        if skip_by_score:
            _set_row_result(idx, 99, 99.0, "99", write_checkpoint=True)
            skipped_by_score_count += 1
            progress.update(1)
            continue

        requested_retries = max_retries
        if retry_policy is not None and "max_retries" in retry_policy:
            requested_retries = int(retry_policy["max_retries"])
        if requested_retries <= 0:
            match_flag, match_conf = _coerce_llm_match_and_confidence("")
            _set_row_result(idx, match_flag, match_conf, "", write_checkpoint=True)
            progress.update(1)
            continue

        left_payload: Dict[str, Any] = {}
        right_payload: Dict[str, Any] = {}

        for col in left_cols:
            col_name = f"{col}{suffixes[0]}" if f"{col}{suffixes[0]}" in candidates.columns else col
            left_payload[col] = None if col_name not in candidates.columns else row[col_name]

        for col in right_cols:
            col_name = f"{col}{suffixes[1]}" if f"{col}{suffixes[1]}" in candidates.columns else col
            right_payload[col] = None if col_name not in candidates.columns else row[col_name]

        user_content = json.dumps({"left": left_payload, "right": right_payload}, default=str)
        llm_inputs[idx] = user_content
        pending_items.append((idx, user_content))

    if pending_items:
        logger.info(
            "[judge_merge_k_out] Starting LLM judging at candidate index "
            f"{pending_items[0][0]} (pending rows: {len(pending_items)})."
        )
    else:
        logger.info("[judge_merge_k_out] No pending rows require LLM judging.")

    def _run_pending(item: Tuple[int, str]) -> Tuple[int, str]:
        idx, user_content = item
        response_text = _judge_once(user_content)
        return idx, response_text

    try:
        if judge_workers == 1:
            for item in pending_items:
                idx = item[0]
                try:
                    _, response_text = _run_pending(item)
                except Exception as judge_runtime_error:
                    _raise_judge_error(judge_runtime_error, stage="judge inference", item_key=idx)
                match_flag, match_conf = _coerce_llm_match_and_confidence(response_text)
                _set_row_result(idx, match_flag, match_conf, response_text, write_checkpoint=True)
                progress.update(1)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=judge_workers) as executor:
                futures = {executor.submit(_run_pending, item): item[0] for item in pending_items}
                for fut in concurrent.futures.as_completed(futures):
                    idx = futures[fut]
                    try:
                        _, response_text = fut.result()
                    except Exception as judge_runtime_error:
                        _raise_judge_error(judge_runtime_error, stage="judge inference", item_key=idx)
                    match_flag, match_conf = _coerce_llm_match_and_confidence(response_text)
                    _set_row_result(idx, match_flag, match_conf, response_text, write_checkpoint=True)
                    progress.update(1)
    except Exception as judge_runtime_error:
        if not isinstance(judge_runtime_error, RuntimeError):
            _raise_judge_error(judge_runtime_error, stage="judge inference")
        raise
    finally:
        progress.close()

    logger.info(
        "[judge_merge_k_out] Summary -> "
        f"pre_judged={existing_judged_count}, resumed={resumed_applied_count}, "
        f"score_skipped={skipped_by_score_count}, llm_judged={len(pending_items)}."
    )

    candidates["llm_is_match"] = llm_matches
    candidates["llm_confidence"] = llm_confidences
    candidates["llm_raw_response"] = llm_raw
    candidates["llm_input_payload"] = llm_inputs

    if confidence_threshold is not None:
        candidates = candidates[candidates["llm_confidence"] >= confidence_threshold]

    return candidates


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
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    failure_log_path: Optional[str] = None,
    retry_policy: Optional[Dict[str, Any]] = None,
    judge_on: Optional[Union[str, List[str]]] = None,
    judge_left_on: Optional[Union[str, List[str]]] = None,
    judge_right_on: Optional[Union[str, List[str]]] = None,
    judge_skip_lower_thresh: Optional[float] = None,
    judge_skip_upper_thresh: Optional[float] = None,
    progress_bar: bool = True,
    judge_workers: int = 1,
    embedding_cache: Optional[str] = None,
) -> DataFrame:
    """
    Retrieve candidate matches with `merge_knn`, then verify each pair with an LLM.
    Adds:
    - `llm_is_match` (0/1)
    - `llm_confidence` (float in [0, 1])
    - `llm_raw_response` (raw model output)

        Judging columns:
        - By default, the judge payload uses `left_on`/`right_on` (or `on`, or shared columns).
        - Override this with `judge_on` for symmetric judging columns, or
            `judge_left_on` / `judge_right_on` for asymmetric column names.

    :param confidence_threshold: optional post-filter threshold on llm_confidence.
    :param judge_skip_lower_thresh: if set, rows with `score` less than or equal to this threshold
        skip LLM judging and are assigned sentinel `99` in llm columns.
    :param judge_skip_upper_thresh: if set, rows with `score` greater than or equal to this threshold
        skip LLM judging and are assigned sentinel `99` in llm columns.
    :param progress_bar: whether to show a progress bar while judging candidate rows.
    :param embedding_cache (str, optional): Folder for an on-disk embedding cache. Embeddings are stored per model and text, so repeated calls only embed texts that are not cached yet. Defaults to None.
    """
    llm_params = llm_params or {}

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

    resolved_judge_model = judge_llm_model
    resolved_provider = llm_provider
    if resolved_provider == "auto":
        resolved_provider = "gemini" if "gemini" in str(resolved_judge_model).lower() else "openai"

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
        embedding_cache=embedding_cache,
    ).copy()

    logger.info(
        "[merge_k_judge] Retrieved candidate set from merge_knn: "
        f"rows={len(candidates)}, k={k}."
    )

    return judge_merge_k_out(
        candidates_df=candidates,
        on=on,
        left_on=left_on,
        right_on=right_on,
        suffixes=suffixes,
        openai_key=openai_key,
        gemini_key=gemini_key,
        judge_llm_model=judge_llm_model,
        llm_provider=llm_provider,
        llm_prompt=llm_prompt,
        llm_params=llm_params,
        max_retries=max_retries,
        ratelimit_sleep_time=ratelimit_sleep_time,
        checkpoint_path=checkpoint_path,
        resume=resume,
        failure_log_path=failure_log_path,
        retry_policy=retry_policy,
        judge_on=judge_on,
        judge_left_on=judge_left_on,
        judge_right_on=judge_right_on,
        judge_skip_lower_thresh=judge_skip_lower_thresh,
        judge_skip_upper_thresh=judge_skip_upper_thresh,
        confidence_threshold=confidence_threshold,
        progress_bar=progress_bar,
        judge_workers=judge_workers,
        failure_function_name="merge_k_judge",
    )


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
    judge_on: Optional[Union[str, List[str]]] = None,
    judge_left_on: Optional[Union[str, List[str]]] = None,
    judge_right_on: Optional[Union[str, List[str]]] = None,
    judge_skip_lower_thresh: Optional[float] = None,
    judge_skip_upper_thresh: Optional[float] = None,
    progress_bar: bool = True,
    judge_workers: int = 1,
    embedding_cache: Optional[str] = None,
) -> DataFrame:
    """Backward-compatible alias for `merge_k_judge`."""
    return merge_k_judge(
        df1=df1,
        df2=df2,
        on=on,
        model=model,
        left_on=left_on,
        right_on=right_on,
        judge_on=judge_on,
        judge_left_on=judge_left_on,
        judge_right_on=judge_right_on,
        judge_skip_lower_thresh=judge_skip_lower_thresh,
        judge_skip_upper_thresh=judge_skip_upper_thresh,
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
        progress_bar=progress_bar,
        judge_workers=judge_workers,
        embedding_cache=embedding_cache,
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
    judge_on: Optional[Union[str, List[str]]] = None,
    judge_left_on: Optional[Union[str, List[str]]] = None,
    judge_right_on: Optional[Union[str, List[str]]] = None,
    judge_skip_lower_thresh: Optional[float] = None,
    judge_skip_upper_thresh: Optional[float] = None,
    progress_bar: bool = True,
    judge_workers: int = 1,
    embedding_cache: Optional[str] = None,
) -> DataFrame:
    """Backward-compatible alias for `merge_k_judge`."""
    return merge_k_judge(
        df1=df1,
        df2=df2,
        on=on,
        model=model,
        left_on=left_on,
        right_on=right_on,
        judge_on=judge_on,
        judge_left_on=judge_left_on,
        judge_right_on=judge_right_on,
        judge_skip_lower_thresh=judge_skip_lower_thresh,
        judge_skip_upper_thresh=judge_skip_upper_thresh,
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
        progress_bar=progress_bar,
        judge_workers=judge_workers,
        embedding_cache=embedding_cache,
    )


def classify_rows(
    df: DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    model: str = None,
    num_labels: int = 2,
    label_map: Optional[dict] = None,
    batch_size: int = 128,
    openai_key: Optional[str] = None,
    gemini_key: Optional[str] = None,
    llm_provider: str = "auto",
    openai_topic: Optional[str] = None,
    openai_prompt: Optional[str] = None,
    openai_params: Optional[dict] = None,
    progress_bar: bool = True,
    api_throttle_seconds: float = 0.0,
    checkpoint_window_size: Optional[int] = None,
    api_workers: int = 1,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    failure_log_path: Optional[str] = None,
    retry_policy: Optional[Dict[str, Any]] = None,
    tokenizer: Optional[str] = None,
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
    :param tokenizer: (str, optional) Tokenizer path or id, for local classifiers saved without tokenizer files.
        Defaults to the model's own tokenizer, or the base model's tokenizer if the checkpoint records it.
    :returns: DataFrame: The dataframe with a new column "clf_preds_{on}" that stores the classification results.
    """

    df = df.copy()
    openai_params = {} if openai_params is None else openai_params

    # load label dict from model path if exists
    if isinstance(model, str) and os.path.exists(os.path.join(model, "label_map.json")):
        with open(os.path.join(model, "label_map.json"), encoding="utf-8") as f:
            label_map = json.load(f)

    # check if label dict is compatible with num of labels
    if label_map is not None and len(label_map) != num_labels:
        raise ValueError(f"label_dict has {label_map} entries, but num_labels is {num_labels}. ")

    # check if the parameters for any inference method is properly specified
    if openai_key is None and gemini_key is None and model is None:
        raise ValueError("Must either specify a model or use an OpenAI key")

    # infer using HF or local models
    if openai_key is None and gemini_key is None:
        import torch

        clf = load_clf(model, num_labels=num_labels)
        tokenizer = _load_tokenizer(model, tokenizer)

        # join texts in columns if needed
        if isinstance(on, list):
            strings_col = serialize_columns(df, on, sep_token=tokenizer.sep_token)
        else:
            strings_col = df[on].tolist()

        # predict in batches (same outputs as transformers' Trainer.predict, without needing accelerate/datasets)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clf.to(device).eval()
        batch_preds = []
        with torch.no_grad():
            for start in tqdm(range(0, len(strings_col), batch_size), disable=not progress_bar):
                batch = tokenizer(strings_col[start:start + batch_size], padding=True, truncation=True, return_tensors="pt")
                logits = clf(**batch.to(device)).logits
                batch_preds.append(logits.argmax(dim=-1).cpu().numpy())
        preds = np.concatenate(batch_preds) if batch_preds else np.array([], dtype=np.int64)

        # convert numeric labels to text labels according to label_map if exists
        if label_map is not None:
            reversed_label_map = {val: key for (key, val) in label_map.items()}
            try:
                preds = [reversed_label_map[pred] for pred in preds]
            except KeyError:
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
        preds = predict_rows_with_llm(
            strings_col,
            model,
            openai_key=openai_key,
            gemini_key=gemini_key,
            llm_provider=llm_provider,
            openai_topic=openai_topic,
            openai_prompt=openai_prompt,
            openai_params=openai_params,
            label_dict=label_map,
            checkpoint_path=checkpoint_path,
            resume=resume,
            failure_log_path=failure_log_path,
            retry_policy=retry_policy,
            batch_size=batch_size,
            throttle_seconds=api_throttle_seconds,
            checkpoint_window_size=checkpoint_window_size,
            workers=api_workers,
            progress_bar=progress_bar,
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
    gemini_key: Optional[str] = None,
    llm_provider: str = "auto",
    model: str = "gpt-4o",
    openai_prompt: Optional[str] = None,
    openai_params: Optional[Dict[str, Any]] = None,
    batch_size: int = 50,
    api_throttle_seconds: float = 0.0,
    checkpoint_window_size: Optional[int] = None,
    api_workers: int = 1,
    output_column: Optional[str] = None,
    progress_bar: bool = True,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    failure_log_path: Optional[str] = None,
    retry_policy: Optional[Dict[str, Any]] = None,
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
    if openai_key is None and gemini_key is None:
        raise ValueError("Must provide `openai_key` or `gemini_key` to use API transforms.")

    if llm_provider not in {"auto", "openai", "gemini"}:
        raise ValueError("llm_provider must be one of {'auto', 'openai', 'gemini'}")

    resolved_provider = llm_provider
    if resolved_provider == "auto":
        if (isinstance(model, str) and "gemini" in model.lower()) or gemini_key is not None:
            resolved_provider = "gemini"
        else:
            resolved_provider = "openai"

    # default prompt
    if openai_prompt is None:
        openai_prompt = (
            "Fix spelling mistakes in each of the following strings. "
        )
    
    logger.info(f"Transforming column(s) {on} with {resolved_provider} model {model} using prompt: {openai_prompt}")

    # serialize multi-column if needed
    if isinstance(on, list):
        tmp_col = "__lt_tmp__"
        df[tmp_col] = serialize_columns(df, on, sep_token=" ")
        column_to_use = tmp_col
        key_name = "-".join(on)
    else:
        column_to_use = on
        key_name = on
    
    null_mask = df[column_to_use].isna()
    df_to_transform = df.loc[~null_mask].copy()

    # final output column name
    out_col = output_column or f"transformed_{key_name}"

    # prepare provider client and fn_kwargs
    if resolved_provider == "openai":
        client = openai.OpenAI(
            api_key=openai_key,
            timeout=openai_params.get("request_timeout", 10)
        )
        transform_fn = openai_transform
        fn_kwargs = {
            "client": client,
            "model": model,
            "prompt": openai_prompt,
            "max_retries": openai_params.get("max_retries", 5),
            "ratelimit_sleep_time": openai_params.get("ratelimit_sleep_time", 15),
            "openai_params": openai_params,
            "failure_log_path": failure_log_path,
            "retry_policy": retry_policy,
        }
    else:
        resolved_gemini_key = gemini_key or os.getenv("GEMINI_API_KEY")
        if resolved_gemini_key is None:
            raise ValueError("Gemini transforms require `gemini_key` or GEMINI_API_KEY.")

        transform_fn = gemini_transform
        fn_kwargs = {
            "gemini_client": _gemini_client(resolved_gemini_key, openai_params.get("request_timeout")),
            "model": model,
            "prompt": openai_prompt,
            "max_retries": openai_params.get("max_retries", 5),
            "ratelimit_sleep_time": openai_params.get("ratelimit_sleep_time", 15),
            "openai_params": openai_params,
            "failure_log_path": failure_log_path,
            "retry_policy": retry_policy,
        }

    # use the generic transform_column util
    result_non_null = transform_column(
        df=df_to_transform,
        column=column_to_use,
        transform_fn=transform_fn,
        fn_kwargs=fn_kwargs,
        chunk_size=batch_size,
        output_column=out_col,
        progress_bar=progress_bar,
        checkpoint_path=checkpoint_path,
        resume=resume,
        failure_log_path=failure_log_path,
        retry_policy=retry_policy,
        throttle_seconds=api_throttle_seconds,
        checkpoint_window_size=checkpoint_window_size,
        workers=api_workers,
    )

    result = df.copy()
    result[out_col] = pd.Series([None] * len(result), index=result.index, dtype="object")
    result.loc[result_non_null.index, out_col] = result_non_null[out_col]

    # clean up temp if used
    if isinstance(on, list):
        result = result.drop(columns=[tmp_col])

    return result
