###Preprocess datasets for training linker
###Prepare data for fine-tuning sbert

import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Union
import pandas as pd

from linktransformer.utils import serialize_columns
from linktransformer.modified_sbert.cluster_fns import clusters_from_edges

def convert_to_text(unicode_string):
    return unicode_string.encode('ascii','ignore').decode('ascii')


def check_clust_data(data,model,text_col_names,clus_id_col_name):

    ###If *_id_name is a string, convert to list
    if isinstance(clus_id_col_name, str):
        clus_id_col_name = [clus_id_col_name]
    
    if isinstance(text_col_names, str):
        text_col_names = [text_col_names]
    
    ### Check if the columns are present in the data or not
    for col in text_col_names:
        if col not in data.columns:
            raise ValueError(f"Column {col} not present in data, please check the left column names")
    ###Check if the clus_id_col_name column is present in the data
    if clus_id_col_name:
        if clus_id_col_name[0] not in data.columns:
            raise ValueError(f"Column {clus_id_col_name} not present in data, please check the label column name")
        
    ### Drop row if all of the left or right columns are empty
    data = data.dropna(subset=text_col_names, how="all")

    # Drop if any id column is missing
    data = data.dropna(subset=clus_id_col_name, how="any")

    ### If clus_id_col_name is not empty, check if it is present in the data
    if clus_id_col_name:
        if clus_id_col_name[0] not in data.columns:
            raise ValueError(f"Column {clus_id_col_name} not present in data, please check the left id column name")
        
    ## Check if text columns form a unique key
    if not data[text_col_names].duplicated().any():
        print(f" Warning text columns do not form a unique key, please check the text column names")
    
    ###Make an idcol as group id
    data["cluster_assignment"] = data.groupby(clus_id_col_name).ngroup().astype(str) + "_g"
    clus_id_col_name="cluster_assignment"

    ## Serialize if there are more than one columns. Feature plan: Later implement deduplication and expansion to avoid rembedding the same string multiple times
    if len(text_col_names) > 1:
        print("Serializing text columns")
        data["text"] = serialize_columns(data, text_col_names, model=model)
    else:
        data["text"] = data[text_col_names[0]]
    return data, clus_id_col_name

    


def check_and_prep_data(data, model,left_col_names, right_col_names, left_id_name, right_id_name, label_col_name):

    ###If *_id_name is a string, convert to list
    if isinstance(left_id_name, str):
        left_id_name = [left_id_name]
    if isinstance(right_id_name, str):
        right_id_name = [right_id_name]

    ### Check if the columns are present in the data or not
    for col in left_col_names:
        if col not in data.columns:
            raise ValueError(f"Column {col} not present in data, please check the left column names")
    
    for col in right_col_names:
        if col not in data.columns:
            raise ValueError(f"Column {col} not present in data, please check the right column names")
        
    ###Check if the label column is present in the data
    if label_col_name:
        if label_col_name not in data.columns:
            raise ValueError(f"Column {label_col_name} not present in data, please check the label column name")
    
    
    ### Drop row if all of the left or right columns are empty
    if  left_col_names:
        data = data.dropna(subset=left_col_names, how="all")
    if  right_col_names:
        data = data.dropna(subset=right_col_names, how="all")

    # Drop if any id column is missing
    if left_id_name:
        data = data.dropna(subset=left_id_name, how="any")
    if right_id_name:
        data = data.dropna(subset=right_id_name, how="any")

    # Drop if any id column is missing
    if left_id_name and right_id_name:
        data = data.dropna(subset=left_id_name+right_id_name, how="any")

    ### If left_id_names is not empty, check if it is present in the data
    if left_id_name:
        if left_id_name[0] not in data.columns:
            raise ValueError(f"Column {left_id_name} not present in data, please check the left id column name")
    
    ### If right_id_name is not empty, check if it is present in the data
    if right_id_name:
        if right_id_name[0] not in data.columns:
            raise ValueError(f"Column {right_id_name} not present in data, please check the right id column name")
    
    ## Check if left columns form a unique key
    if not data[left_col_names].duplicated().any():
        print(f" Warning Left columns do not form a unique key, please check the left column names")
    
    #### Check if right columns form a unique key, leave a warning if not and drop duplicates
    if data[right_col_names].duplicated().any():
        print("Warning: Right columns do not form a unique key, dropping duplicates. Matching will proceed")
        data = data.drop_duplicates(subset=right_col_names)

    ## If left_id_name is not specified, we will assume left data is unique and use the index as the id
    if not left_id_name:
        data["left_id"] =  data.groupby(left_col_names).ngroup().astype(str) + "_l"
        left_id_name = "left_id"
        print("Warning: left id column not specified, using the left columns to group the data and using the groupby index as the id")
    else:
        ### If left id is specified, we can group by the left id column and use the groupby index as the id
        data["left_id"] = data.groupby(left_id_name).ngroup().astype(str) + "_l" ## Convert to string
        left_id_name = "left_id"
    
    ### If right id is not specified, we can group by the right columns and use the groupby index as the id
    if not right_id_name:
        data["right_id"] = data.groupby(right_col_names).ngroup().astype(str) + "_r"
        right_id_name = "right_id"
        print("Warning: right id column not specified, using the right columns to group the data and using the groupby index as the id")

    else:
        ## If right id is specified, we can group by the right id column and use the groupby index as the id
        data["right_id"] = data.groupby(right_id_name).ngroup().astype(str) + "_r"
        right_id_name = "right_id"

    
    ## Serialize if there are more than one columns. Feature plan: Later implement deduplication and expansion to avoid rembedding the same string multiple times
    if len(left_col_names) > 1:
        print("Serializing left columns")
        data["left_text"] = serialize_columns(data, left_col_names, model=model)
    else:
        data["left_text"] = data[left_col_names[0]]
    
    if len(right_col_names) > 1:
        data["right_text"] = serialize_columns(data, right_col_names, model=model)
    else:
        data["right_text"] = data[right_col_names[0]]
    return data, left_id_name, right_id_name



def preprocess_any_data(data: Union[str, pd.DataFrame]=None,
                        train_data: Union[str, pd.DataFrame] = None,
                        val_data: Union[str, pd.DataFrame] = None,
                        test_data: Union[str, pd.DataFrame] = None,
        left_col_names: List[str] = None,
        right_col_names: List[str] = None,
        left_id_name: List[str] = None,
        right_id_name: List[str] = None,
        label_col_name: str = None,
        clus_id_col_name: str = None,
        clus_text_col_names: List[str] = None,
        model: str = "sentence-transformers/all-mpnet-base-v2",
        val_perc: float = 0.2,
        val_query_prop:float =0.5,
        large_val: bool = True,
        test_at_end=True
                        ):
    """
    Wrapper around the two preprocessing functions. if label_col_name is not none, use prep_paired_label_data; otherwise use prep_linkage_data

    :param: data : Path to the csv or excel file or dataframe
    :param: left_col_names: List of column names to use for linkage in the left dataframe. Each list is a set of columns that are used to make a pair. Left columns need to be unique.
    :param: right_col_names: List of column names to use for linkage in the right dataframe. Each list is a set of columns that are used to make a pair. Right columns can have duplicates.
    :param: left_id_name: List of column names to use as id for the left columns.
    :param: right_id_name: List of column names to use as id for the right columns.
    :param label_col_name: Name of the column to use as label (0 or 1). Specify if you have a label column
    :param clusterid_col_name: Name of the column to use as cluster id. Specify if you have a cluster id column
    :param cluster_text_col_name: Name of the column to use as text for the cluster. Specify if you have a cluster text column
    :param: model: Model to use for tokenization.
    :param: val_perc: Percentage of data to use for validation. Defaults to 0.2.
    :param: large_val: If True, use the training data as part of the validation data (in the corpus for information retrieval evaluation)
    :return: A tuple containing the training data dictionary and the validation data, test tuple.
    """
    #Validate inputs
    if label_col_name and clus_id_col_name:
        raise ValueError("Please specify either label_col_name or clusterid_col_name. Not both")
    elif label_col_name and clus_text_col_names:
        raise ValueError("Please specify either label_col_name or cluster_text_col_name. Not both")
    elif clus_id_col_name and not clus_text_col_names:
        raise ValueError("Please specify cluster_text_col_name if you specify clusterid_col_name")
    elif clus_text_col_names and not clus_id_col_name:
        raise ValueError("Please specify clusterid_col_name if you specify cluster_text_col_name")
        

    if label_col_name:
        return prep_paired_label_data(
            data = data,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            left_col_names=left_col_names,
            right_col_names=right_col_names,
            left_id_name=left_id_name,
            right_id_name=right_id_name,
            label_col_name=label_col_name,
            model=model,
            val_perc=val_perc,
            test_at_end=test_at_end
        )
    elif clus_id_col_name and clus_text_col_names:
        return prep_clus_data(
            data = data,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            text_col_names=clus_text_col_names,
            clus_id_col_name=clus_id_col_name,
            model=model,
            val_perc=val_perc,
            val_query_prop=val_query_prop,
            test_at_end=test_at_end,
            large_val=large_val
        )


    else:
        return prep_linkage_data(
            data = data,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            left_col_names=left_col_names,
            right_col_names=right_col_names,
            left_id_name=left_id_name,
            right_id_name=right_id_name,
            model=model,
            val_perc=val_perc,
            large_val=large_val,
            test_at_end=test_at_end
        )
    
    


def prep_paired_label_data(
        data = None,
        train_data=None,
        val_data=None,
        test_data=None,
        left_col_names: List[str] = [],
        right_col_names: List[str] = [],
        left_id_name: List[str] = [],
        right_id_name: List[str] = [],
        label_col_name: str = "label",
        model: str = "sentence-transformers/all-mpnet-base-v2",
        val_perc: float = 0.2,
        test_at_end=True
) ->  Tuple[Dict[str, List[str]], Tuple[List[str], List[str], List[str]],Tuple[List[str], List[str], List[str]]]:
    """
    This method is used to prepare training data + evaluation data. Evaluation is suitable for classification evaluation.
    What it would give for the eval data is a tuple of sentences1, sentences2, labels. 

    :param: data : Path to the csv or excel file or dataframe
    :param: train_data: Path to the csv or excel file or dataframe for training data
    :param: val_data: Path to the csv or excel file or dataframe for validation data
    :param: test_data: Path to the csv or excel file or dataframe for test data
    :param: left_col_names: List of column names to use for linkage in the left dataframe. Each list is a set of columns that are used to make a pair. Left columns need to be unique.
    :param: right_col_names: List of column names to use for linkage in the right dataframe. Each list is a set of columns that are used to make a pair. Right columns can have duplicates.
    :param: left_id_name: List of column names to use as id for the left columns.
    :param: right_id_name: List of column names to use as id for the right columns.
    :param label_col_name: Name of the column to use as label (0 or 1)
    :param: model: Model to use for tokenization.
    :param: val_perc: Percentage of data to use for validation. Defaults to 0.2.
    :return: A tuple containing the training data dictionary and the validation data, test tuple.
    """

    ###Check that only data or all of train, val, test are specified
    if data is not None and (train_data is not None or val_data is not None or test_data is not None):
        raise ValueError("Please specify either data or train, val, test data. Not both")
    elif data is  None and  (train_data is  None and val_data is  None and test_data is None):
        raise ValueError("Please specify either data or train, val, test data. Not none")
    elif data is not None:
        ### Load the data if xlsx, else csv
        if isinstance(data, pd.DataFrame):
            data = data
        elif data.endswith(".xlsx"):
            data = pd.read_excel(data)
        elif data.endswith(".csv"):
            data = pd.read_csv(data)
        else:
            raise ValueError("Data should be a path to a csv or excel file or a dataframe")
    ##Reset indices of data
        data=data.reset_index(drop=True)

    elif train_data is not None and val_data is not None and test_data is not None:
        ### Load the data if xlsx, else csv
        if isinstance(train_data, pd.DataFrame):
            train_data = train_data
        elif train_data.endswith(".xlsx"):
            train_data = pd.read_excel(train_data)
        elif train_data.endswith(".csv"):
            train_data = pd.read_csv(train_data)
        else:
            raise ValueError("Data should be a path to a csv or excel file or a dataframe")

        if isinstance(val_data, pd.DataFrame):
            val_data = val_data
        elif val_data.endswith(".xlsx"):
            val_data = pd.read_excel(val_data)
        elif val_data.endswith(".csv"):
            val_data = pd.read_csv(val_data)
        else:
            raise ValueError("Data should be a path to a csv or excel file or a dataframe")

        if isinstance(test_data, pd.DataFrame):
            test_data = test_data
        elif test_data.endswith(".xlsx"):
            test_data = pd.read_excel(test_data)
        elif test_data.endswith(".csv"):
            test_data = pd.read_csv(test_data)
        else:
            raise ValueError("Data should be a path to a csv or excel file or a dataframe")
        
        ##Reset indices of train, val, test data
        train_data=train_data.reset_index(drop=True)
        val_data=val_data.reset_index(drop=True)
        test_data=test_data.reset_index(drop=True)
        
        data=train_data


    data = data.copy()

    data, left_id_rename, right_id_rename = check_and_prep_data(data, model, left_col_names, right_col_names, left_id_name, right_id_name, label_col_name)
    ### We now want to make a network from this data. Simply use the left_id and right_id columns as edges.
    ### Once we do that, we want to make connected components from this network. Each connected component is a cluster that is used as a "class" for training.
    edge_list = list(zip(data[left_id_rename], data[right_id_rename]))
    cluster_assignment = clusters_from_edges(edge_list)
    ## Assign cluster ids to each node
    ## Now, make a mapping between left_id and cluster id - reverse the dict
    cluster_assignment = {k: v for v, l in cluster_assignment.items() for k in l}
    data["cluster_assignment"] = data[left_id_rename].map(cluster_assignment)

    ### We want to split the data by cluster assignment 
    if val_perc == 1:
        train_data = data
        val_data = data
        ## Throw warning that train=val data
        print("Warning: train and val data are the same")
    else:
        if  val_data is None  and  test_data is None:
        ### Split by val perc - we want to split by cluster assignment. Only 20% of the clusters should be in the val set
            train_cluster_assignment, val_cluster_assignment = train_test_split(list(set(data["cluster_assignment"])), test_size=val_perc, random_state=42)
            ###Split val into test and val
            if test_at_end:
                print("Splitting val into test and val (equally) ")
                val_cluster_assignment, test_cluster_assignment = train_test_split(val_cluster_assignment, test_size=0.5, random_state=42)
                train_data = data[data["cluster_assignment"].isin(train_cluster_assignment)]
                val_data = data[data["cluster_assignment"].isin(val_cluster_assignment)]
                test_data = data[data["cluster_assignment"].isin(test_cluster_assignment)]
            else:
                train_data = data[data["cluster_assignment"].isin(train_cluster_assignment)]
                val_data = data[data["cluster_assignment"].isin(val_cluster_assignment)]
            
        else:
            val_data,left_id_rename, right_id_rename = check_and_prep_data(val_data,model, left_col_names, right_col_names, left_id_name, right_id_name, label_col_name)
            test_data,left_id_rename, right_id_rename = check_and_prep_data(test_data, model, left_col_names, right_col_names, left_id_name, right_id_name, label_col_name)
            
    train_data=data
    ### Now, group by cluster assignment and make a dict with cluster_assignment:[left_text, right_text1, right_text2, right_text3...]
    train_data_dict = defaultdict(list)
    for index, row in train_data.iterrows():
        train_data_dict[row["cluster_assignment"]].append(row["left_text"])
        train_data_dict[row["cluster_assignment"]].append(row["right_text"])          
    
    ### Deduplicate the lists
    train_data_dict = {k: list(set(v)) for k, v in train_data_dict.items()}

    ###Time for validation set now.
    ###Simply make a list of left_text, right_text1, right_text2, right_text3... and make a list of labels
    val_left_text_list = []
    val_right_text_list = []
    val_labels_list = []

    for index, row in val_data.iterrows():
        val_left_text_list.append(row["left_text"])
        val_right_text_list.append(row["right_text"])
        val_labels_list.append(row[label_col_name])

    ###Time for test set now.
    if test_at_end:
    ###Simply make a list of left_text, right_text1, right_text2, right_text3... and make a list of labels
        test_left_text_list = []
        test_right_text_list = []
        test_labels_list = []

        for index, row in test_data.iterrows():
            test_left_text_list.append(row["left_text"])
            test_right_text_list.append(row["right_text"])
            test_labels_list.append(row[label_col_name])
    else:
        test_left_text_list, test_right_text_list, test_labels_list=None,None,None

    return train_data_dict, (val_left_text_list, val_right_text_list, val_labels_list), (test_left_text_list, test_right_text_list, test_labels_list)    



def prep_linkage_data(
    data: Union[str, pd.DataFrame] = None,
    train_data: Union[str, pd.DataFrame] =None,
    val_data: Union[str, pd.DataFrame] =None,
    test_data: Union[str, pd.DataFrame] =None,
    left_col_names: List[str] = [],
    right_col_names: List[str] = [],
    left_id_name: List[str] = [],
    right_id_name: List[str] = [],
    model: str = "sentence-transformers/all-mpnet-base-v2",
    val_perc: float = 0.2,
    large_val: bool = True,
    test_at_end=True
) -> Tuple[Dict[str, List[str]], Tuple[Dict[str, str], Dict[str, str], Dict[str, set]],Tuple[Dict[str, str], Dict[str, str], Dict[str, set]]] :
    """
    Process linkage data to create training and validation sets. In this method, we get validation data suitable for information retrieval evaluation.
    What that means is that things queries are looked for in a corpus - an information retrieval task. 

    Args:
        data (str or df): Path to the csv or excel file or dataframe
        train_data (str or df): Path to the csv or excel file or dataframe for training data
        val_data (str or df): Path to the csv or excel file or dataframe for validation data
        test_data (str or df): Path to the csv or excel file or dataframe for test data
        left_col_names (List[str]): List of column names to use for linkage in the left dataframe. Each list is a set of columns that are used to make a pair. Left columns need to be unique.
        right_col_names (List[str]): List of column names to use for linkage in the right dataframe. Each list is a set of columns that are used to make a pair. Right columns can have duplicates.
        left_id_name (List[str]): List of column names to use as id for the left columns.
        right_id_name (List[str]): List of column names to use as id for the right columns.
        model (str): Model to use for tokenization.
        val_perc (float): Percentage of data to use for validation. Defaults to 0.2.
        large_val (bool): If True, use the training data as part of the validation data.

    Returns:
        Tuple[Dict[str, List[str]], Tuple[Dict[str, str], Dict[str, str], Dict[str, set]]]: A tuple containing the training data dictionary and the validation data tuple.

    """


    ###Check that only data or all of train, val, test are specified
    if data is not None and (train_data is not None or val_data is not None or test_data is not None):
        raise ValueError("Please specify either data or train, val, test data. Not both")
    elif data is  None and  (train_data is  None and val_data is  None and test_data is None):
        raise ValueError("Please specify either data or train, val, test data. Not none")
    elif data is not None:
        ### Load the data if xlsx, else csv
        if isinstance(data, pd.DataFrame):
            data = data
        elif data.endswith(".xlsx"):
            data = pd.read_excel(data)
        elif data.endswith(".csv"):
            data = pd.read_csv(data)
        else:
            raise ValueError("Data should be a path to a csv or excel file or a dataframe")
    ##Reset indices of data
        data=data.reset_index(drop=True)

    elif train_data is not None and val_data is not None and test_data is not None:
        ### Load the data if xlsx, else csv
        if isinstance(train_data, pd.DataFrame):
            train_data = train_data
        elif train_data.endswith(".xlsx"):
            train_data = pd.read_excel(train_data)
        elif train_data.endswith(".csv"):
            train_data = pd.read_csv(train_data)
        else:
            raise ValueError("Data should be a path to a csv or excel file or a dataframe")

        if isinstance(val_data, pd.DataFrame):
            val_data = val_data
        elif val_data.endswith(".xlsx"):
            val_data = pd.read_excel(val_data)
        elif val_data.endswith(".csv"):
            val_data = pd.read_csv(val_data)
        else:
            raise ValueError("Data should be a path to a csv or excel file or a dataframe")

        if isinstance(test_data, pd.DataFrame):
            test_data = test_data
        elif test_data.endswith(".xlsx"):
            test_data = pd.read_excel(test_data)
        elif test_data.endswith(".csv"):
            test_data = pd.read_csv(test_data)
        else:
            raise ValueError("Data should be a path to a csv or excel file or a dataframe")
        
        ##Reset indices of train, val, test data
        train_data=train_data.reset_index(drop=True)
        val_data=val_data.reset_index(drop=True)
        test_data=test_data.reset_index(drop=True)
        
        data=train_data

    data = data.copy()

    data, left_id_rename, right_id_rename = check_and_prep_data(data, model, left_col_names, right_col_names, left_id_name, right_id_name, label_col_name=None)
    
    ### We now want to make a network from this data. Simply use the left_id and right_id columns as edges.
    ### Once we do that, we want to make connected components from this network. Each connected component is a cluster that is used as a "class" for training.
    edge_list = list(zip(data[left_id_rename], data[right_id_rename]))
    cluster_assignment = clusters_from_edges(edge_list)
    ## Assign cluster ids to each node
    ## Now, make a mapping between left_id and cluster id - reverse the dict
    cluster_assignment = {k: v for v, l in cluster_assignment.items() for k in l}
    data["cluster_assignment"] = data[left_id_rename].map(cluster_assignment)


    train_data=data
    ### We want to split the data by cluster assignment 
    if val_perc == 1:
        train_data = data
        val_data = data
        ## Throw warning that train=val data
        print("Warning: train and val data are the same")
    else:
        if  val_data is None  and  test_data is None:
            ### Split by val perc - we want to split by cluster assignment. Only 20% of the clusters should be in the val set
            train_cluster_assignment, val_cluster_assignment = train_test_split(list(set(data["cluster_assignment"])), test_size=val_perc, random_state=42)
            
            ###Split val into test and val
            if test_at_end:
                print("Splitting val into test and val (equally) ")
                val_cluster_assignment, test_cluster_assignment = train_test_split(val_cluster_assignment, test_size=0.5, random_state=42)
                train_data = data[data["cluster_assignment"].isin(train_cluster_assignment)]
                val_data = data[data["cluster_assignment"].isin(val_cluster_assignment)]
                test_data = data[data["cluster_assignment"].isin(test_cluster_assignment)]
            else:
                train_data = data[data["cluster_assignment"].isin(train_cluster_assignment)]
                val_data = data[data["cluster_assignment"].isin(val_cluster_assignment)]
            
        else:
            val_data,left_id_rename, right_id_rename = check_and_prep_data(val_data,model, left_col_names, right_col_names, left_id_name, right_id_name, label_col_name)
            
            if test_data is not None:
                test_data,left_id_rename, right_id_rename = check_and_prep_data(test_data, model, left_col_names, right_col_names, left_id_name, right_id_name, label_col_name)
            

    ### Now, group by cluster assignment and make a dict with cluster_assignment:[left_text, right_text1, right_text2, right_text3...]
    train_data_dict = defaultdict(list)
    for index, row in train_data.iterrows():
        train_data_dict[row["cluster_assignment"]].append(row["left_text"])
        train_data_dict[row["cluster_assignment"]].append(row["right_text"])          
    
    ### Deduplicate the lists
    train_data_dict = {k: list(set(v)) for k, v in train_data_dict.items()}

    queries = {}
    corpus = {}
    relevant_docs = defaultdict(set)

    for index, row in val_data.iterrows():
        queries[row[left_id_rename]] = row["left_text"]
        corpus[row[right_id_rename]] = row["right_text"]
        relevant_docs[row[left_id_rename]].add(row[right_id_rename])

    ## If large_val is True, add the training data to the validation data corpus
    if large_val:
        ##First, need to rename the 
        for index, row in train_data.iterrows():
            corpus[row[right_id_rename]+"_train"] = row["right_text"]
            relevant_docs[row[left_id_rename]+"_train"].add(row[right_id_rename]+"_train")

    val_data = queries, corpus, relevant_docs ## Needed for the evaluation function

    ##Now, prepare the test data
    if test_at_end:
        queries = {}
        corpus = {}
        relevant_docs = defaultdict(set)

        for index, row in test_data.iterrows():
            queries[row[left_id_rename]] = row["left_text"]
            corpus[row[right_id_rename]] = row["right_text"]
            relevant_docs[row[left_id_rename]].add(row[right_id_rename])
        
        test_data = queries, corpus, relevant_docs
    else:
        test_data = None


    return train_data_dict, val_data, test_data
    

def preprocess_mexican_tarrif_data(file_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/wiki_data/es_mexican_products.xlsx"):
    ##Load the df
    df=pd.read_excel(file_path)
    df=df.copy()
    ##Drop missing descriptions
    df=df.dropna(subset=["description47","description48"])
    ##Make descriptions lowercase
    df=df.applymap(lambda x: x.lower() if isinstance(x,str) else x)
    ##Drop dropna rows
    df=df.dropna()
    ##Get the left and right ids
    df1=df[["tariffcode47","description47"]]
    ###Add a ground truth column
    df1["ground_truth"]=df["description48"]
    ##Print duplcated rows
    df2=df[["tariffcode48","description48"]]
    ##Drop duplicates from df1
    df1=df1.drop_duplicates(subset=["tariffcode47","description47"])
    df1=df1.drop_duplicates(subset=["description47"])

    df1.to_csv("df1.csv",index=False)
    df2.to_csv("df2.csv",index=False)

    return df1,df2


def prep_clus_data(
    data: Union[str, pd.DataFrame] = None,
    train_data: Union[str, pd.DataFrame] =None,
    val_data: Union[str, pd.DataFrame] =None,
    test_data: Union[str, pd.DataFrame] =None,
    text_col_names: List[str] = None,
    clus_id_col_name: List[str] = None,
    model: str = "sentence-transformers/all-mpnet-base-v2",
    val_perc: float = 0.2,
    val_query_prop: float = 0.5,
    large_val: bool = True,
    test_at_end=True
) -> Tuple[Dict[str, List[str]], Tuple[Dict[str, str], Dict[str, str], Dict[str, set]],Tuple[Dict[str, str], Dict[str, str], Dict[str, set]]] :
    """

    """
    
    ###Check that only data or all of train, val, test are specified
    if data is not None and (train_data is not None or val_data is not None or test_data is not None):
        raise ValueError("Please specify either data or train, val, test data. Not both")
    elif data is  None and  (train_data is  None and val_data is  None and test_data is None):
        raise ValueError("Please specify either data or train, val, test data. Not none")
    elif data is not None:
        ### Load the data if xlsx, else csv
        if isinstance(data, pd.DataFrame):
            data = data
        elif data.endswith(".xlsx"):
            data = pd.read_excel(data)
        elif data.endswith(".csv"):
            data = pd.read_csv(data)
        else:
            raise ValueError("Data should be a path to a csv or excel file or a dataframe")
    ##Reset indices of data
        data=data.reset_index(drop=True)

    elif train_data is not None and val_data is not None and test_data is not None:
        ### Load the data if xlsx, else csv
        if isinstance(train_data, pd.DataFrame):
            train_data = train_data
        elif train_data.endswith(".xlsx"):
            train_data = pd.read_excel(train_data)
        elif train_data.endswith(".csv"):
            train_data = pd.read_csv(train_data)
        else:
            raise ValueError("Data should be a path to a csv or excel file or a dataframe")

        if isinstance(val_data, pd.DataFrame):
            val_data = val_data
        elif val_data.endswith(".xlsx"):
            val_data = pd.read_excel(val_data)
        elif val_data.endswith(".csv"):
            val_data = pd.read_csv(val_data)
        else:
            raise ValueError("Data should be a path to a csv or excel file or a dataframe")

        if isinstance(test_data, pd.DataFrame):
            test_data = test_data
        elif test_data.endswith(".xlsx"):
            test_data = pd.read_excel(test_data)
        elif test_data.endswith(".csv"):
            test_data = pd.read_csv(test_data)
        else:
            raise ValueError("Data should be a path to a csv or excel file or a dataframe")
        
        ##Reset indices of train, val, test data
        train_data=train_data.reset_index(drop=True)
        val_data=val_data.reset_index(drop=True)
        test_data=test_data.reset_index(drop=True)
        
        data=train_data

    data = data.copy()

    data,cluster_id_rename=check_clust_data(data, model, text_col_names, clus_id_col_name)
    
    
    train_data=data
    ### We want to split the data by cluster assignment 
    if val_perc == 1:
        train_data = data
        val_data = data
        ## Throw warning that train=val data
        print("Warning: train and val data are the same")
    else:
        if  val_data is None  and  test_data is None:
            ### Split by val perc - we want to split by cluster assignment. Only 20% of the clusters should be in the val set
            train_cluster_assignment, val_cluster_assignment = train_test_split(list(set(data["cluster_assignment"])), test_size=val_perc, random_state=42)
            
            ###Split val into test and val
            if test_at_end:
                print("Splitting val into test and val (equally) ")
                val_cluster_assignment, test_cluster_assignment = train_test_split(val_cluster_assignment, test_size=0.5, random_state=42)
                train_data = data[data["cluster_assignment"].isin(train_cluster_assignment)]
                val_data = data[data["cluster_assignment"].isin(val_cluster_assignment)]
                test_data = data[data["cluster_assignment"].isin(test_cluster_assignment)]
            else:
                train_data = data[data["cluster_assignment"].isin(train_cluster_assignment)]
                val_data = data[data["cluster_assignment"].isin(val_cluster_assignment)]
            
        else:
            val_data,cluster_id_rename  = check_clust_data(val_data, model, text_col_names, clus_id_col_name)
    
            if test_data is not None:
                test_data,cluster_id_rename = check_clust_data(test_data, model, text_col_names, clus_id_col_name)
    

    ### Now, group by cluster assignment and make a dict with cluster_assignment:[left_text, right_text1, right_text2, right_text3...]
    # print("Preparing train data")
    # train_data_dict = defaultdict(list)
    # for index, row in train_data.iterrows():
    #     train_data_dict[row["cluster_assignment"]].append(row["text"])
    
    # ### Deduplicate the lists
    # train_data_dict = {k: list(set(v)) for k, v in train_data_dict.items()}
    train_data_dict = train_data.groupby("cluster_assignment")["text"].apply(set).apply(list).to_dict()

    print("Preparing val data")

    queries = {}
    corpus = {}
    relevant_docs = defaultdict(set)

    ###We want 10% of the cluster to form queries, the rest corpus. 
    ###We should iterate over groups for this
    for cluster_id, group in val_data.groupby(cluster_id_rename):
        query_portion=int(len(group)*val_query_prop)
        query_group=group.iloc[:query_portion].reset_index(drop=True)
        corpus_group=group.iloc[query_portion:].reset_index(drop=True)
        query_group_ids=query_group[cluster_id_rename].tolist()
        query_group_ids=[query_group_ids[i]+str(i) for i in range(len(query_group_ids))]

        corpus_group_ids=corpus_group[cluster_id_rename].tolist()
        corpus_group_ids=[corpus_group_ids[i]+str(i)+"c" for i in range(len(corpus_group_ids))]

        ##Relevant docs contain all corpus ids for each query
        for qindex, query_id in enumerate(query_group_ids):
            queries[query_id]=query_group["text"].iloc[qindex]
            relevant_docs[query_id]=set(corpus_group_ids)

        for cindex,corpus_id in enumerate(corpus_group_ids):
            corpus[corpus_id]=corpus_group["text"].iloc[cindex]




    if large_val:
        ##ADd train data to the corpus 
        for cluster_id, group in train_data.groupby(cluster_id_rename):
            group=group.reset_index(drop=True)
            corpus_group_ids=group[cluster_id_rename].tolist()
            corpus_group_ids=[corpus_group_ids[i]+str(i)+"ct" for i in range(len(corpus_group_ids))]


            for cindex,corpus_id in enumerate(corpus_group_ids):
                corpus[corpus_id]=group["text"].iloc[cindex]

    val_data = queries, corpus, relevant_docs ## Needed for the evaluation function

    ##Now, prepare the test data
    if test_at_end:
        print("Preparing test data")

        queries = {}
        corpus = {}
        relevant_docs = defaultdict(set)
        for cluster_id, group in test_data.groupby(cluster_id_rename):
            query_portion=int(len(group)*val_query_prop)
            query_group=group.iloc[:query_portion]
            corpus_group=group.iloc[query_portion:]
            query_group_ids=query_group[cluster_id_rename].tolist()
            query_group_ids=[query_group_ids[i]+str(i) for i in range(len(query_group_ids))]
            ###Add an index after each query_group_id
            # for index,group_id in enumerate(query_group_ids) :
            #     query_group_ids[index]=query_group_ids[index]+"_"+str(index)
            
            corpus_group_ids=corpus_group[cluster_id_rename].tolist()
            corpus_group_ids=[corpus_group_ids[i]+str(i)+"c" for i in range(len(corpus_group_ids))]

            ##Relevant docs contain all corpus ids for each query
            for qindex, query_id in enumerate(query_group_ids):
                queries[query_id]=query_group["text"].iloc[qindex]
                relevant_docs[query_id]=set(corpus_group_ids)

            for cindex,corpus_id in enumerate(corpus_group_ids):
                corpus[corpus_id]=corpus_group["text"].iloc[cindex]
            
        test_data = queries, corpus, relevant_docs
    else:
        test_data = None


    return train_data_dict, val_data, test_data
    






##Run as script
if __name__ == "__main__":
   
    # train_data,val_data=prep_linkage_data("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/wiki_data/es_mexican_products.xlsx",left_col_names=["description47"],right_col_names=['description48'],left_id_name=['tariffcode47'],right_id_name=['tariffcode48'],model="all-mpnet-base-v2",val_perc=0.2,large_val=False)

    # train_data,val_data,test_data=prep_clus_data(data="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/src/linktransformer/data/company_clusters.csv",text_col_names=["company_name"],clus_id_col_name=["cluster_id"],model="all-mpnet-base-v2",val_perc=0.2,large_val=False)
    # # train_data,val_data,test_data=preprocess_any_data(data="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/src/linktransformer/data/company_clusters.csv",text_col_names=["company_name"],clus_id_col_name=["cluster_id"],model="all-mpnet-base-v2",val_perc=0.2,large_val=False)
    # print(train_data)
    # print(val_data)
    # print(test_data)
    # train_data,val_data,test_data=preprocess_any_data(data="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/src/linktransformer/data/company_clusters.csv",clus_text_col_names=["company_name"],clus_id_col_name=["cluster_id"],model="all-mpnet-base-v2",val_perc=0.2,large_val=True)
    # print(train_data)
    # print(val_data)
    # print(test_data)
    train_data,val_data,test_data=preprocess_any_data(data="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/src/linktransformer/data/es_mexican_products.xlsx",
                                                      left_col_names=["description47"],
                                                        right_col_names=['description48'],
                                                        left_id_name=['tariffcode47'],
                                                        right_id_name=['tariffcode48'],
                                                      model="all-mpnet-base-v2",
                                                      val_perc=0.2,large_val=True)
    print(train_data)
    print(val_data)
    print(test_data)