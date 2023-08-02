import pandas as pd 
import numpy as np
import os
from sklearn.metrics import f1_score
# import hyperopt

from linktransformer.infer import lm_merge
from hyperopt import fmin, tpe, hp
from time import time
from linktransformer.train_model import train_model
import json

def infer_deepmatcher_data(data_dir:str="Beer",model="all-mpnet-base-v2",on=['title','manufacturer','price'],merge_type="1:m"):
    """
    There are 5 files for each dataset in the DeepMatcher repo. tableA.csv  tableB.csv  test.csv  train.csv  valid.csv
    tableA and tableB need to be linked. train and valid are the training and validation sets respectively and we test on test.
    """

    ###Test zero-shot merge using the all-mpnet-base-v2 model
    ###Load the data
    tableA = pd.read_csv(os.path.join(data_dir, "tableA.csv"))
    tableB = pd.read_csv(os.path.join(data_dir, "tableB.csv"))

    ##Rename the "id" variable to id_dm
    tableA = tableA.rename(columns={"id":"id_dm"})
    tableB = tableB.rename(columns={"id":"id_dm"})
    
    ###Subset the data using the anno files
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "valid.csv"))

    ##Subset table A and B for the test and validation sets
    table_A_test = tableA[tableA.id_dm.isin(test_df.ltable_id)]
    table_B_test = tableB[tableB.id_dm.isin(test_df.rtable_id)]

    table_A_val = tableA[tableA.id_dm.isin(val_df.ltable_id)]
    table_B_val = tableB[tableB.id_dm.isin(val_df.rtable_id)]

    #Drop duplicates in A val and test
    # table_A_test = table_A_test.drop_duplicates(subset=on)
    # table_A_val = table_A_val.drop_duplicates(subset=on)


    ###Now, link the tables using the all-mpnet-base-v2 model
    merged_df_val = lm_merge(table_A_val, table_B_val, model=model,on=on,merge_type=merge_type)
    merged_df_test = lm_merge(table_A_test, table_B_test, model=model,on=on,merge_type=merge_type)

    ##Rename idx and idy to ltable_id and rtable_id
    merged_df_val = merged_df_val.rename(columns={"id_dm_x":"ltable_id","id_dm_y":"rtable_id"})
    merged_df_test = merged_df_test.rename(columns={"id_dm_x":"ltable_id","id_dm_y":"rtable_id"})

    ###Now, link the validation set to the merged df using [idx,idy] as the left keys and ltable_id,rtable_id as the right keys
    merged_df_val = merged_df_val.merge(val_df, on=["ltable_id","rtable_id"], how="right")
    merged_df_test = merged_df_test.merge(test_df, on=["ltable_id","rtable_id"], how="right")

    def calculate_f1(threshold):
        merged_df_val["predicted"] = np.where(merged_df_val["score"] > threshold, 1, 0)
        ##Replace the value of predicted label with 0 if the score is NaN
        merged_df_val["predicted"] = np.where(merged_df_val["score"].isna(), 0, merged_df_val["predicted"])
        ##Now, calculate the f1 score
        f1 = f1_score(merged_df_val["label"], merged_df_val["predicted"])
        print(f"F1 score on the validation set with threshold {threshold} is {f1}")
        return -f1
    
     # Hyperopt optimization to find the best threshold for F1
    space = hp.uniform('threshold', 0, 1)
    best = fmin(fn=calculate_f1, space=space, algo=tpe.suggest, max_evals=1000)
    best_threshold = best['threshold']
    print(f"Best threshold: {best_threshold}")

    ###Using the best threshold, calculate F1 on the test set
    start_time = time()
    merged_df_test["predicted"] = np.where(merged_df_test["score"] > best_threshold, 1, 0)
    ##Replace the value of predicted label with 0 if the score is NaN
    merged_df_test["predicted"] = np.where(merged_df_test["score"].isna(), 0, merged_df_test["predicted"])
    ##Now, calculate the f1 score on the test set
    test_f1 = f1_score(merged_df_test["label"], merged_df_test["predicted"])
    print(f"F1 score on the test set with threshold {best_threshold} is {test_f1}")
    end_time = time()
    return test_f1 , end_time-start_time, best_threshold

 

def prep_train_lt_model_on_deepmatcher_data(data_dir:str="Beer"):
    """
    There are 5 files for each dataset in the DeepMatcher repo. tableA.csv  tableB.csv  test.csv  train.csv  valid.csv
    tableA and tableB need to be linked. train and valid are the training and validation sets respectively and we test on test.
    """

    ###Test zero-shot merge using the all-mpnet-base-v2 model
    ###Load the data
    tableA = pd.read_csv(os.path.join(data_dir, "tableA.csv"))
    ##Rename id as id_dm_x
    tableA = tableA.rename(columns={"id":"id_dm_x"})

    tableB = pd.read_csv(os.path.join(data_dir, "tableB.csv"))
    ##Rename id as id_dm_y
    tableB = tableB.rename(columns={"id":"id_dm_y"})

    ##GEt only the train sets
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    ##Keep only positive examples
    train_df = train_df[train_df.label==1]
    ##Also get the validation set
    val_df = pd.read_csv(os.path.join(data_dir, "valid.csv"))
    ##Keep only positive examples
    val_df = val_df[val_df.label==1]
    train_df = pd.concat([train_df,val_df],axis=0)
    ###Keeping both in train to be in line with LT API
    ###rename ltable_id and rtable_id to id_dm_x and id_dm_y
    train_df = train_df.rename(columns={"ltable_id":"id_dm_x","rtable_id":"id_dm_y"})
    ##Merge with tableA and tableB
    train_df = train_df.merge(tableA, on="id_dm_x", how="left")
    train_df = train_df.merge(tableB, on="id_dm_y", how="left")
    ##WE now have a dataframe of positive pairs
    return train_df






if __name__ == "__main__":

    ###Make a dict of datasets and their match keys
    deep_matcher_datasets={
        "Structured/Amazon-Google":{},
        "Structured/Beer":{},
        "Structured/DBLP-ACM":{},
        "Structured/DBLP-GoogleScholar":{},
        "Structured/iTunes-Amazon":{},
        "Structured/Walmart-Amazon":{},
        "Structured/Fodors-Zagats":{},
        }
    pretrained_model="multi-qa-MiniLM-L6-cos-v1"
    
    ##Add pretrained model to the dict to all datasets
    for dataset in deep_matcher_datasets:
        deep_matcher_datasets[dataset]["base_model"]=pretrained_model
    ###Iterate through all the datasets and get the match keys and add them to the dict as "on"

    for dataset in deep_matcher_datasets:
        print("Preparing to train on dataset: ",dataset)
        ###Load the data
        tableA = pd.read_csv(os.path.join(dataset, "tableA.csv"))
        tableB = pd.read_csv(os.path.join(dataset, "tableB.csv"))
        ###Get the match keys
        match_keys = list(set(tableA.columns).intersection(set(tableB.columns)))
        ##Remove id from key
        match_keys=[key for key in match_keys if key!="id"]
        print("Match keys are: ",match_keys)

        ###Add them to the dict
        deep_matcher_datasets[dataset]["on"]=match_keys

    ##Check zero shot performance on all the datasets
    for dataset in deep_matcher_datasets.keys():
        print(deep_matcher_datasets[dataset]["on"])
        test_f1 , time_taken, best_threshold = infer_deepmatcher_data(dataset,model=deep_matcher_datasets[dataset]["base_model"],on=deep_matcher_datasets[dataset]["on"],merge_type="1:m")
        print(f"Test F1 on {dataset} is {test_f1} and time taken is {time_taken} and best threshold is {best_threshold}")
        ##Add the stats to the dict
        deep_matcher_datasets[dataset]["test_f1_zs"]=test_f1
        deep_matcher_datasets[dataset]["time_taken_zs"]=time_taken
        deep_matcher_datasets[dataset]["best_threshold_zs"]=best_threshold

    ###Now, iterate through all the datasets and train a model on them
    for dataset in deep_matcher_datasets.keys():
        train_data=prep_train_lt_model_on_deepmatcher_data(dataset)
        ##ADd _x and _y to the match keys
        ##REmove id from the match keys
        on=deep_matcher_datasets[dataset]["on"]
        on_keys_left = [key+"_x" for key in on]
        on_keys_right = [key+"_y" for key in on]
        ##Train a model on this data
        saved_model_path = train_model(
            model_path = deep_matcher_datasets[dataset]["base_model"],
            ##Data can be path to the excel file or a dataframe
            data =train_data,
            left_col_names= on_keys_left ,
            right_col_names= on_keys_right,
            left_id_name= ['id_dm_x'],
            right_id_name= ['id_dm_y'],
            training_args = {"num_epochs":10,"model_save_name":f"linkage_model_{dataset}_{deep_matcher_datasets[dataset]['base_model']}"}
        )
        print(saved_model_path)
    
    ###Now, iterate through all the datasets and infer on them
        test_f1 , time_taken, best_threshold = infer_deepmatcher_data(dataset,model=saved_model_path,on=deep_matcher_datasets[dataset]["on"],merge_type="1:m")
        print(f"Test F1 on {dataset} is {test_f1} and time taken is {time_taken} and best threshold is {best_threshold}")
        ##Add the stats to the dict
        deep_matcher_datasets[dataset]["test_f1"]=test_f1
        deep_matcher_datasets[dataset]["time_taken"]=time_taken
        deep_matcher_datasets[dataset]["best_threshold"]=best_threshold

    ##Save the dict
    with open("deepmatcher_results.json","w") as f:
        json.dump(deep_matcher_datasets,f)

    
    #     


        
    
    # deep_matcher_dataset="Structured/iTunes-Amazon/"
    # training_val_data=prep_train_lt_model_on_deepmatcher_data("Structured/iTunes-Amazon/",on=["Song_Name","Artist_Name","Album_Name"])

    # ##Train a model on this data
    # saved_model_path = train_model(
    #     model_path = "multi-qa-MiniLM-L6-cos-v1",
    #     ##Data can be path to the excel file or a dataframe
    #     data =None,
    #     left_col_names= ["description47"],
    #     right_col_names= ['description48'],
    #     left_id_name= ['tariffcode47'],
    #     right_id_name= ['tariffcode48'],
    #     training_args = {"num_epochs":10}
    # )
    # print(saved_model_path)

    # ##Now, infer on the test set
    # test_f1 , time_taken, best_threshold = infer_deepmatcher_data("Structured/iTunes-Amazon/",model="all-mpnet-base-v2",on=['title','manufacturer','price'],merge_type="1:m")







