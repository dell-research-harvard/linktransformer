import pandas as pd 
import numpy as np
import os
from sklearn.metrics import f1_score
# import hyperopt

from linktransformer.infer import  evaluate_pairs
from hyperopt import fmin, tpe, hp
from time import time
from linktransformer.train_model import train_model
import json
import wandb


 
def evaluate_deep_matcher_data(data_dir,model,left_on,right_on,note):
    """Calculate pair-wise cosine similarity between the left and right columns of the data and evaluate on the test set."""
    tableA = pd.read_csv(os.path.join(data_dir, "tableA.csv")) ##_tok for company name
    tableB = pd.read_csv(os.path.join(data_dir, "tableB.csv"))

    ##Rename the "id" variable to id_dm
    tableA = tableA.rename(columns={"id":"ltable_id"})
    tableB = tableB.rename(columns={"id":"rtable_id"})

    ##load test and val
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "valid.csv"))

    ###Merge tableA and tableB with the test and val sets sequentially to get the test and val sets with the left and right columns labels
    full_test_df = test_df.merge(tableA, on="ltable_id", how="left").merge(tableB, on="rtable_id", how="left")
    full_val_df = val_df.merge(tableA, on="ltable_id", how="left").merge(tableB, on="rtable_id", how="left")

    print("Columns in the test set are: ",full_test_df.columns)

    ##Now, calculate the cosine similarity between the left and right columns
    full_test_df = evaluate_pairs(full_test_df, left_on=left_on, right_on=right_on, model=model)
    full_val_df = evaluate_pairs(full_val_df, left_on=left_on, right_on=right_on, model=model)

    ###Now, we have the cosine similarity scores for the test and val sets. We want to tune the threshold on the val set and then evaluate on the test set
    def calculate_f1(threshold):
        full_val_df["predicted"] = np.where(full_val_df["score"] > threshold, 1, 0)
        ##Now, calculate the f1 score
        f1 = f1_score(full_val_df["label"], full_val_df["predicted"],)
        return -f1    

    # Hyperopt optimization to find the best threshold for F1
    space = hp.uniform('threshold', 0, 1)
    best = fmin(fn=calculate_f1, space=space, algo=tpe.suggest, max_evals=1000, verbose=False)
    best_threshold = best['threshold']

    ###Using the best threshold, calculate F1 on the test set
    start_time = time()
    full_test_df["predicted"] = np.where(full_test_df["score"] > best_threshold, 1, 0)
    ##Replace the value of predicted label with 0 if the score is NaN
    full_test_df["predicted"] = np.where(full_test_df["score"].isna(), 0, full_test_df["predicted"])
    ##Now, calculate the f1 score on the test set (after dropping na)
    test_f1 = f1_score(full_test_df["label"], full_test_df["predicted"])

    print(f"F1 score on the test set with threshold {best_threshold} is {test_f1}")
    end_time = time()
    return test_f1 , end_time-start_time, best_threshold, full_test_df




def prep_train_lt_model_on_deepmatcher_data_only_positives(data_dir:str="Beer",combine_train_val:bool=False):
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

    test_df=pd.read_csv(os.path.join(data_dir, "test.csv"))
    ##Keep only positive examples
    test_df = test_df[test_df.label==1]
    ###Keeping both in train to be in line with LT API
    ###rename ltable_id and rtable_id to id_dm_x and id_dm_y
    train_df = train_df.rename(columns={"ltable_id":"id_dm_x","rtable_id":"id_dm_y"})
    ##Merge with tableA and tableB
    train_df = train_df.merge(tableA, on="id_dm_x", how="left")
    train_df = train_df.merge(tableB, on="id_dm_y", how="left")

    ##prep val and test
    val_df=val_df.rename(columns={"ltable_id":"id_dm_x","rtable_id":"id_dm_y"})
    val_df=val_df.merge(tableA, on="id_dm_x", how="left")
    val_df=val_df.merge(tableB, on="id_dm_y", how="left")

    test_df=test_df.rename(columns={"ltable_id":"id_dm_x","rtable_id":"id_dm_y"})
    test_df=test_df.merge(tableA, on="id_dm_x", how="left")
    test_df=test_df.merge(tableB, on="id_dm_y", how="left")

    if combine_train_val:
        train_df=pd.concat([train_df,val_df],axis=0)
        val_df=None


    return train_df,val_df,test_df


def prep_train_lt_model_on_deepmatcher_data_paired(data_dir:str="Beer",combine_train_val:bool=False):
    """
    This variant would retain deepmatcher datatset's original structure with tuples of (left, right, label) and would not be in line with the LT API
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
    ###Keeping both in train to be in line with LT API
    ###rename ltable_id and rtable_id to id_dm_x and id_dm_y
    train_df = train_df.rename(columns={"ltable_id":"id_dm_x","rtable_id":"id_dm_y"})
    ##Merge with tableA and tableB
    train_df = train_df.merge(tableA, on="id_dm_x", how="left")
    train_df = train_df.merge(tableB, on="id_dm_y", how="left")
    
    val_df=val_df.rename(columns={"ltable_id":"id_dm_x","rtable_id":"id_dm_y"})
    val_df=val_df.merge(tableA, on="id_dm_x", how="left")
    val_df=val_df.merge(tableB, on="id_dm_y", how="left")

    test_df=pd.read_csv(os.path.join(data_dir, "test.csv"))
    test_df=test_df.rename(columns={"ltable_id":"id_dm_x","rtable_id":"id_dm_y"})
    test_df=test_df.merge(tableA, on="id_dm_x", how="left")
    test_df=test_df.merge(tableB, on="id_dm_y", how="left")

    if combine_train_val:
        train_df=pd.concat([train_df,val_df],axis=0)
        val_df=None

    return train_df,val_df,test_df
   



def run_train_eval_retrieval_dm_datasets(deep_matcher_datasets,pretrained_model):
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
        on=deep_matcher_datasets[dataset]["on"]
        on_keys_left = [key+"_x" for key in on]
        on_keys_right = [key+"_y" for key in on]
        test_f1, time_taken, best_threshold,merged_df_test = evaluate_deep_matcher_data(dataset,model=deep_matcher_datasets[dataset]["base_model"],left_on=on_keys_left,right_on=on_keys_right,note="zs")
        ##Add the stats to the dict as zs stats
        deep_matcher_datasets[dataset]["zs_test_f1"]=test_f1
        deep_matcher_datasets[dataset]["zs_time_taken"]=time_taken
        deep_matcher_datasets[dataset]["zs_best_threshold"]=best_threshold
        merged_df_test.to_csv(f"{dataset}/zs_deepmatcher_results.csv",index=False)
        

    ###Now, iterate through all the datasets and train a model on them
    for dataset in deep_matcher_datasets.keys():
        train_data,val_data, test_data=prep_train_lt_model_on_deepmatcher_data_only_positives(dataset,combine_train_val=False)
        train_data.to_csv(f"{dataset}/train_data_processed.csv",index=False)
        val_data.to_csv(f"{dataset}/val_data_processed.csv",index=False)
        test_data.to_csv(f"{dataset}/test_data_processed.csv",index=False)
        # print(val_data)
        # print(train_data)
        ##ADd _x and _y to the match keys
        ##REmove id from the match keys
        on=deep_matcher_datasets[dataset]["on"]
        on_keys_left = [key+"_x" for key in on]
        on_keys_right = [key+"_y" for key in on]
        #Train a model on this data
        saved_model_path = train_model(
            model_path = deep_matcher_datasets[dataset]["base_model"],
            ##Data can be path to the excel file or a dataframe
            data=None,
            train_data =train_data,
            val_data=val_data,
            test_data=test_data,
            left_col_names= on_keys_left ,
            right_col_names= on_keys_right,
            left_id_name= ['id_dm_x'],
            right_id_name= ['id_dm_y'],
            training_args = {"num_epochs":100,"test_at_end":True,"model_save_name":f"linkage_model_retrieval_supcon_{dataset}",
                             "wandb_names":{
                                     "id": "econabhishek",
                                    "run": f"linkage_model_retrieval_supcon_{dataset}_{deep_matcher_datasets[dataset]['base_model']}",
                                    "project": "benchmark_data_set_ret",
                                    "entity": "econabhishek" },
                             "learning_rate": 2e-5,
                             "warmup_perc": 1,
                            "large_val":True
            },
            log_wandb=True
        )
        test_f1, time_taken, best_threshold,merged_df_test = evaluate_deep_matcher_data(dataset,model=saved_model_path,left_on=on_keys_left,right_on=on_keys_right,note="zs")
        print(f"Test F1 on {dataset} is {test_f1} and time taken is {time_taken} and best threshold is {best_threshold}")
        ##Add the stats to the dict
        deep_matcher_datasets[dataset]["test_f1"]=test_f1
        deep_matcher_datasets[dataset]["time_taken"]=time_taken
        deep_matcher_datasets[dataset]["best_threshold"]=best_threshold
        deep_matcher_datasets[dataset]["model_path"]=saved_model_path
        merged_df_test.to_csv(f"{dataset}/trained_deepmatcher_results_retreival_supcon.csv",index=False)

    ##Save the dict
    with open(f"retrieval_benchmark_results_retrieval_supcon.json","w") as f:
        json.dump(deep_matcher_datasets,f)


def run_train_eval_classfication_dm_datasets(deep_matcher_datasets,pretrained_model,loss_type="supcon",learning_rate=2e-5):
    
    
    ##Add pretrained model to the dict to all datasets
    for dataset in deep_matcher_datasets:
        deep_matcher_datasets[dataset]["base_model"]=pretrained_model

    ###Iterate through all the datasets and get the match keys and add them to the dict as "on"

    for dataset in deep_matcher_datasets:
        dataset_name=dataset.split("/")[-2:]
        dataset_name="_".join(dataset_name)
        dataset_name=dataset_name+pretrained_model+loss_type
        print("Preparing to train on dataset: ",dataset_name)
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
    for dataset in list(deep_matcher_datasets.keys()):
        on=deep_matcher_datasets[dataset]["on"]
        on_keys_left = [key+"_x" for key in on]
        on_keys_right = [key+"_y" for key in on]
        test_f1, time_taken, best_threshold,merged_df_test = evaluate_deep_matcher_data(dataset,model=deep_matcher_datasets[dataset]["base_model"],left_on=on_keys_left,right_on=on_keys_right,note="zs")
        ##Add the stats to the dict as zs stats
        deep_matcher_datasets[dataset]["zs_test_f1"]=test_f1
        deep_matcher_datasets[dataset]["zs_time_taken"]=time_taken
        deep_matcher_datasets[dataset]["zs_best_threshold"]=best_threshold
        merged_df_test.to_csv(f"{dataset}/zs_deepmatcher_results_{os.path.basename(pretrained_model)}_{str(learning_rate)}.csv",index=False)
    

    ###Now, iterate through all the datasets and train a model on them
    for dataset in list(deep_matcher_datasets.keys()):
        train_data,val_data, test_data=prep_train_lt_model_on_deepmatcher_data_paired(dataset,combine_train_val=False)
        train_data.to_csv(f"{dataset}/train_data_processed.csv",index=False)
        val_data.to_csv(f"{dataset}/val_data_processed.csv",index=False)
        test_data.to_csv(f"{dataset}/test_data_processed.csv",index=False)
        # print(val_data)
        # print(train_data)
        ##ADd _x and _y to the match keys
        ##REmove id from the match keys
        on=deep_matcher_datasets[dataset]["on"]
        on_keys_left = [key+"_x" for key in on]
        on_keys_right = [key+"_y" for key in on]
        #Train a model on this data
        saved_model_path = train_model(
            model_path = deep_matcher_datasets[dataset]["base_model"],
            ##Data can be path to the excel file or a dataframe
            data=None,
            train_data =train_data,
            val_data=val_data,
            test_data=test_data,
            left_col_names= on_keys_left ,
            right_col_names= on_keys_right,
            left_id_name= ['id_dm_x'],
            right_id_name= ['id_dm_y'],
            label_col_name="label",
            training_args = {"num_epochs":75,"test_at_end":True,
                             "model_save_name":f"linkage_model_{loss_type}_paired_{dataset_name}_{str(learning_rate)}",
                             "batch_size": 128,
                             "learning_rate": learning_rate,
                             "warmup_perc": 1,
                             "eval_steps_perc":0.5,
                             "wandb_names":{
                                     "id": "econabhishek",
                                    "run": f"linkage_model_{dataset_name}_{loss_type}_paired_{str(learning_rate)}",
                                    "project": "benchmark_data_set_class",
                                    "entity": "econabhishek" },
                            "large_val":False
            },
            log_wandb=True
        )
        test_f1, time_taken, best_threshold,merged_df_test = evaluate_deep_matcher_data(dataset,model=saved_model_path,left_on=on_keys_left,right_on=on_keys_right,note="zs")
        print(f"Test F1 on {dataset} is {test_f1} and time taken is {time_taken} and best threshold is {best_threshold}")
        ##Add the stats to the dict
        deep_matcher_datasets[dataset]["test_f1"]=test_f1
        deep_matcher_datasets[dataset]["time_taken"]=time_taken
        deep_matcher_datasets[dataset]["best_threshold"]=best_threshold
        deep_matcher_datasets[dataset]["model_path"]=saved_model_path
        merged_df_test.to_csv(f"{dataset}/trained_deepmatcher_results_paired_{loss_type}_{os.path.basename(pretrained_model)}_{str(learning_rate)}.csv",index=False)



    ##Save the dict
    with open(f"class_benchmark_results_{loss_type}_{os.path.basename(pretrained_model)}_{str(learning_rate)}.json","w") as f:
        json.dump(deep_matcher_datasets,f)







if __name__ == "__main__":
    
    data_dir="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/deepmatcher_data"

    ###Make a dict of datasets and their match keys
    deep_matcher_datasets={
        # "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/deepmatcher_data/Structured/Amazon-Google":{},
        # "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/deepmatcher_data/Structured/Beer":{},
        # "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/deepmatcher_data/Structured/DBLP-ACM":{},
        # "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/deepmatcher_data/Structured/DBLP-GoogleScholar":{},
        # "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/deepmatcher_data/Structured/iTunes-Amazon":{},
        # "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/deepmatcher_data/Structured/Walmart-Amazon":{},
        # "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/deepmatcher_data/Structured/Fodors-Zagats":{},
        # "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/deepmatcher_data/Dirty/DBLP-ACM" : {},
        # "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/deepmatcher_data/Dirty/DBLP-GoogleScholar" : {},
        # "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/deepmatcher_data/Dirty/iTunes-Amazon" : {},
        # "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/deepmatcher_data/Dirty/Walmart-Amazon" : {},
        # "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/deepmatcher_data/Textual/Abt-Buy":{},
        "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/deepmatcher_data/Textual/Company":{} ## To do Follows a diff naming convention

        }

    pretrained_model= 'multi-qa-mpnet-base-dot-v1' #"BAAI/bge-large-en-v1.5" #"BAAI/bge-large-en-v1.5"  #"sentence-transformers/all-mpnet-base-v2" #BAAI/bge-large-zh-v1.5 #multi-qa-mpnet-base-dot-v1
    # run_train_eval_retrieval_dm_datasets(deep_matcher_datasets,pretrained_model)
    # run_train_eval_classfication_dm_datasets(deep_matcher_datasets,pretrained_model,loss_type="supcon")


    run_train_eval_classfication_dm_datasets(deep_matcher_datasets,pretrained_model,loss_type="supcon",learning_rate=2e-5)

    # evaluate_deep_matcher_data("Textual/Company",model=pretrained_model,left_on=["content_x"],right_on=["content_y"],note="zs")