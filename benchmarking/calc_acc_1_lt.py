import linktransformer as lt
import pandas as pd
import pickle
import numpy as np
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




def calculate_retrieval_accuracy(path_to_pickle):
    with open(path_to_pickle, 'rb') as handle:
        val_data = pickle.load(handle)

    # Merge queries and corpus using LinkTransformer's lt.merge function
    merged_df = lt.merge(pd.DataFrame.from_dict(val_data[0], orient='index', columns=['query_text']).reset_index(), 
                         pd.DataFrame.from_dict(val_data[1], orient='index', columns=['corpus_text']).reset_index(), 
                         merge_type='1:m', model="text-embedding-ada-002", 
                         left_on="query_text", right_on="corpus_text", openai_key="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")
    
    merged_df=merged_df.reset_index()

    merged_df.to_csv("merged_df.csv")

    print(merged_df.columns)
    print(merged_df.head())
    # # Create a new column in merged_df that indicates whether each document (cid) is relevant to its corresponding query (qid)
    def is_relevant(row):
        return 1 if row['index_y'] in val_data[2][row['index_x']] else 0

    merged_df['is_relevant'] = merged_df.apply(is_relevant, axis=1)

    # Calculate the retrieval accuracy by taking the mean of the is_relevant column
    accuracy = merged_df['is_relevant'].mean()

    return accuracy


 
def evaluate_deep_matcher_data_gpt(val_pickle,test_pickle,model,left_on,right_on,label_col,openai_key=None):

    ##Load the data
    with open(val_pickle, 'rb') as handle:
        val_data = pickle.load(handle)
    with open(test_pickle, 'rb') as handle:
        test_data = pickle.load(handle)

    ###EEach pikcle was a tuple - (left_text,right_text,label)
    ###Convert to a dataframe
    print(val_data[0])

    val_df=pd.Dataframe(val_data,columns=[left_on,right_on,label_col])
    print(val_df.head())

    val_df = pd.DataFrame(val_data,columns=[left_on,right_on,label_col])
    test_df = pd.DataFrame(test_data,columns=[left_on,right_on,label_col])


    print("Columns in the test set are: ",full_test_df.columns)

    ##Now, calculate the cosine similarity between the left and right columns
    full_test_df = lt.evaluate_pairs(val_df, left_on=left_on, right_on=right_on, model=model,openai_key=openai_key)
    full_val_df = lt.evaluate_pairs(test_df, left_on=left_on, right_on=right_on, model=model,openai_key=openai_key)

    ###Now, we have the cosine similarity scores for the test and val sets. We want to tune the threshold on the val set and then evaluate on the test set
    def calculate_f1(threshold):
        full_val_df["predicted"] = np.where(full_val_df["score"] > threshold, 1, 0)
        ##Now, calculate the f1 score
        f1 = f1_score(full_val_df["label"], full_val_df["predicted"],)
        return -f1    

    # Hyperopt optimization to find the best threshold for F1
    space = hp.uniform('threshold', 0, 1)
    best = fmin(fn=calculate_f1, space=space, algo=tpe.suggest, max_evals=10000, verbose=False)
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


    



