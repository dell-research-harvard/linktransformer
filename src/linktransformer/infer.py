###Inference and Linkage script
###We want to link dfs together using embeddings
import numpy as np
import pandas as pd
import faiss
from typing import Union, List, Optional, Tuple,Dict, Any
from pandas import DataFrame

from linktransformer.modified_sbert.cluster_fns import cluster
from linktransformer.utils import serialize_columns, infer_embeddings, load_model, cosine_similarity







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
    cosine_similarity_12 = cosine_similarity(embeddings1,embeddings2)
    print(cosine_similarity_12.shape)

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
    print(f"Deduplicating dataframe with originally {len(df)} rows")

    df = df.copy()

    ### First, deduplicate based on exact matches
    df = df.drop_duplicates(subset=on, keep="first")
    print(f"Exact matches found: dropping them")
    print(f"Number of rows after exact match deduplication: {len(df)}")

    ### Now, deduplicate based on similarity threshold
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

    print(f"Deduplicating dataframe with originally {len(df)} rows")
    df = cluster_rows(df, model, on, cluster_type, cluster_params, openai_key)
    df = df.drop_duplicates(subset="cluster", keep="first")
    df = df.drop(columns=["cluster"])
    print(f"Number of rows after deduplication: {len(df)}")

    return df


    

    


