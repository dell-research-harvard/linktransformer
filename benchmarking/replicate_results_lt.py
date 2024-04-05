import linktransformer as lt
from rapidfuzz import process, fuzz

import pandas as pd
import pickle
import numpy as np
import pandas as pd 
import numpy as np
import os
from sklearn.metrics import f1_score
# import hyperopt
from concurrent.futures import ProcessPoolExecutor, as_completed

from linktransformer.infer import  evaluate_pairs
from hyperopt import fmin, tpe, hp
from time import time
import json
import wandb
import editdistance
from tqdm import tqdm


from huggingface_hub import hf_hub_download



def calculate_lev_distance(str_1,str_2):
    return editdistance.eval(str_1,str_2)

def calculate_batch_distances(query, corpus_texts):
    # Using RapidFuzz to find the best match for a query within the corpus texts
    # Note: process.extractOne returns the best match, its score, and its index
    best_match = process.extractOne(query, corpus_texts, scorer=fuzz.ratio)
    return best_match[2], best_match[1]  # Returning index and score (edit distance) of the best match

def calculate_retrieval_accuracy_edit(path_to_pickle):
    with open(path_to_pickle, 'rb') as handle:
        val_data = pickle.load(handle)

    query_df = pd.DataFrame.from_dict(val_data[0], orient='index', columns=['query_text']).reset_index()
    corpus_df = pd.DataFrame.from_dict(val_data[1], orient='index', columns=['corpus_text']).reset_index()
    
    corpus_texts = corpus_df['corpus_text'].tolist()

    # Prepare for parallel computation with a progress bar
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(calculate_batch_distances, query, corpus_texts): query for query in query_df['query_text']}
        
        distances = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating distances"):
            distances.append(future.result())

    # Convert distances to a DataFrame
    matches_df = pd.DataFrame(distances, columns=['corpus_index', 'score'], index=query_df['index'])
    matches_df['corpus_index'] = matches_df['corpus_index'].apply(lambda x: corpus_df.iloc[x]['index'])

    # Add relevance information
    matches_df['is_relevant'] = matches_df.apply(lambda row: 1 if row['corpus_index'] in val_data[2].get(row.name, []) else 0, axis=1)

    # Calculate the retrieval accuracy
    accuracy = matches_df['is_relevant'].mean()

    return accuracy

def eval_pairs_using_edit_distance(df, left_on="left_text", right_on="right_text"):
    ###If left_on or right_on len > 1, concatenate the columns

    ###now, calculate edit distance for each left_on , right on row and add a "score" column
    df["score"]=df.apply(lambda x: calculate_lev_distance(x[left_on],x[right_on]),axis=1)

    return df


def calculate_total_queries_in_split(path_to_pickle):
    with open(path_to_pickle, 'rb') as handle:
        val_data = pickle.load(handle)

        query_df=pd.DataFrame.from_dict(val_data[0], orient='index', columns=['query_text']).reset_index()
        print("query_df shape",query_df.shape)

    return len(query_df)

def make_val_test_query_size_table(model_dict):
    key_names=list(model_dict.keys())
    table=pd.DataFrame(columns=["key","val_size","test_size"])
    for key in key_names:
        lt_model_path=model_dict[key]["LT"]
        val_pickle=os.path.join(lt_model_path,"val_data.pickle")
        test_pickle=os.path.join(lt_model_path,"test_data.pickle")
        val_size=calculate_total_queries_in_split(val_pickle)
        test_size=calculate_total_queries_in_split(test_pickle)
        table=pd.concat([table,pd.DataFrame([[key,val_size,test_size]],columns=["key","val_size","test_size"])],ignore_index=True)
    return table



def calculate_retrieval_accuracy_gpt(path_to_pickle,openai_key,openai_model="text-embedding-3-small"):
    with open(path_to_pickle, 'rb') as handle:
        val_data = pickle.load(handle)

    # Merge queries and corpus using LinkTransformer's lt.merge function
    merged_df = lt.merge(pd.DataFrame.from_dict(val_data[0], orient='index', columns=['query_text']).reset_index(), 
                         pd.DataFrame.from_dict(val_data[1], orient='index', columns=['corpus_text']).reset_index(), 
                         merge_type='1:m', model=openai_model, 
                         left_on="query_text", right_on="corpus_text", openai_key=openai_key)
    

    merged_df=merged_df.reset_index()

  
    # # Create a new column in merged_df that indicates whether each document (cid) is relevant to its corresponding query (qid)
    def is_relevant(row):
        return 1 if row['index_y'] in val_data[2][row['index_x']] else 0

    merged_df['is_relevant'] = merged_df.apply(is_relevant, axis=1)

    # Calculate the retrieval accuracy by taking the mean of the is_relevant column
    accuracy = merged_df['is_relevant'].mean()

    return accuracy


def calculate_retrieval_accuracy_lt(path_to_pickle,model):
    with open(path_to_pickle, 'rb') as handle:
        val_data = pickle.load(handle)

    # Merge queries and corpus using LinkTransformer's lt.merge function
    merged_df = lt.merge(pd.DataFrame.from_dict(val_data[0], orient='index', columns=['query_text']).reset_index(), 
                         pd.DataFrame.from_dict(val_data[1], orient='index', columns=['corpus_text']).reset_index(), 
                         merge_type='1:m', model=model, 
                         left_on="query_text", right_on="corpus_text", openai_key=None)
    
    print("merged_df shape",merged_df.shape)
    merged_df=merged_df.reset_index()



    # # Create a new column in merged_df that indicates whether each document (cid) is relevant to its corresponding query (qid)
    def is_relevant(row):
        return 1 if row['index_y'] in val_data[2][row['index_x']] else 0

    merged_df['is_relevant'] = merged_df.apply(is_relevant, axis=1)

    merged_df.to_csv("merged_df.csv")

    # Calculate the retrieval accuracy by taking the mean of the is_relevant column
    accuracy = merged_df['is_relevant'].mean()

    return accuracy

def make_table(model_dict, openai_key):
    """This function takes a dictionary of models (sbert , lt) and returns a table with edit distance, zs sbert, LT, gpt (ada)"""

    table=pd.DataFrame(columns=["key","edit_distance","sbert","lt","gpt_small","gpt_large"])
    

    for key in model_dict.keys():
        sbert_model=model_dict[key]["SBERT"]
        lt_model=model_dict[key]["LT"]
        
        path_to_pickle=hf_hub_download(repo_id=lt_model, filename="test_data.pickle")
        # path_to_pickle=os.path.join(lt_model,"test_data.pickle")
        sbert_accuracy=calculate_retrieval_accuracy_lt(path_to_pickle,sbert_model)
        lt_accuracy=calculate_retrieval_accuracy_lt(path_to_pickle,lt_model)
        if openai_key is not None:
            gpt_accuracy_small=calculate_retrieval_accuracy_gpt(path_to_pickle,openai_key,openai_model="text-embedding-3-small")
            gpt_accuracy_large=calculate_retrieval_accuracy_gpt(path_to_pickle,openai_key,openai_model="text-embedding-3-large")
        else:
            gpt_accuracy_small=None
            gpt_accuracy_large=None
        editdistance_acc=calculate_retrieval_accuracy_edit(path_to_pickle)
        print("Edit distance accuracy",editdistance_acc)
        ##Use pd.concat to add a row to the table
        table=pd.concat([table,pd.DataFrame([[key,editdistance_acc,sbert_accuracy,lt_accuracy,gpt_accuracy_small,gpt_accuracy_large]],columns=["key","edit_distance","sbert","lt","gpt_small","gpt_large"])],ignore_index=True)
        print(table)

    return table


def make_mexican_table(model_dict, openai_key,openai_model="text-embedding-3-small"):
    """This function takes a dictionary of models (sbert , lt) and returns a table with edit distance, zs sbert, LT, gpt (ada)"""

    table=pd.DataFrame(columns=["key","edit_distance","sbert","lt","gpt"])
    

    for key in model_dict.keys():
        sbert_model=model_dict[key]["SBERT"]
        lt_model=model_dict[key]["LT"]
        lt_un_model=model_dict[key]["LT_UN"]
        
        # hf_hub_download(repo_id=lt_model, filename="test_data.pickle")
        path_to_pickle=os.path.join(lt_model,"val_data.pickle")
        sbert_accuracy=calculate_retrieval_accuracy_lt(path_to_pickle,sbert_model)
        lt_un_accuracy=calculate_retrieval_accuracy_lt(path_to_pickle,lt_un_model)
        lt_accuracy=calculate_retrieval_accuracy_lt(path_to_pickle,lt_model)
        if openai_key is not None:
            gpt_accuracy_small=calculate_retrieval_accuracy_gpt(path_to_pickle,openai_key,openai_model="text-embedding-3-small")
            gpt_accuracy_large=calculate_retrieval_accuracy_gpt(path_to_pickle,openai_key,openai_model="text-embedding-3-large")
        else:
            gpt_accuracy_small=None
            gpt_accuracy_large=None
        editdistance_acc=calculate_retrieval_accuracy_edit(path_to_pickle)

        ##Use pd.concat to add a row to the table
        table=pd.concat([table,pd.DataFrame([[key,editdistance_acc,sbert_accuracy,lt_un_accuracy,lt_accuracy,gpt_accuracy_small,gpt_accuracy_large]],columns=["key","edit_distance","sbert","lt_un","lt","gpt_small","gpt_large"])],ignore_index=True)
        print(table)
    return table    

 
def evaluate_f1_score(val_pickle,test_pickle,model,openai_key=None,edit_distance=False):
    """This function needs some work - it is not complete yet"""
    print("Using model",model)
    ##Load the data
    with open(val_pickle, 'rb') as handle:
        val_data = pickle.load(handle)
    with open(test_pickle, 'rb') as handle:
        test_data = pickle.load(handle)

    ###EEach pikcle was a tuple - (left_text,right_text,label)
    ###Convert to a dataframe
    ##Structure (val_left_text_list, val_right_text_list, val_labels_list), (test_left_text_list, test_right_text_list, test_labels_list)   
    val_df=pd.DataFrame({"left_text":val_data[0],"right_text":val_data[1],"label":val_data[2]})
    test_df=pd.DataFrame({"left_text":test_data[0],"right_text":test_data[1],"label":test_data[2]})



    ##Now, calculate the cosine similarity between the left and right columns
    if not edit_distance:
        full_test_df = lt.evaluate_pairs(test_df, left_on="left_text", right_on="right_text", model=model,openai_key=openai_key)
        full_val_df = lt.evaluate_pairs(val_df, left_on="left_text", right_on="right_text", model=model,openai_key=openai_key)
    else:
        full_test_df = eval_pairs_using_edit_distance(test_df, left_on="left_text", right_on="right_text")
        full_val_df = eval_pairs_using_edit_distance(val_df, left_on="left_text", right_on="right_text")

    ###Now, we have the cosine similarity scores for the test and val sets. We want to tune the threshold on the val set and then evaluate on the test set
    def calculate_f1(threshold):
        full_val_df["predicted"] = np.where(full_val_df["score"] > threshold, 1, 0)
        ##Now, calculate the f1 score
        f1 = f1_score(full_val_df["label"], full_val_df["predicted"],)
        return -f1    

    # Hyperopt optimization to find the best threshold for F1
    space = hp.uniform('threshold', 0.5,1)
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


 

def get_size_of_japanese_data(val_pickle,test_pickle):
    ##Load the data
    with open(val_pickle, 'rb') as handle:
        val_data = pickle.load(handle)
    with open(test_pickle, 'rb') as handle:
        test_data = pickle.load(handle)

    ###EEach pikcle was a tuple - (left_text,right_text,label)
    ###Convert to a dataframe
    ##Structure (val_left_text_list, val_right_text_list, val_labels_list), (test_left_text_list, test_right_text_list, test_labels_list)   
    val_df=pd.DataFrame({"left_text":val_data[0],"right_text":val_data[1],"label":val_data[2]})
    test_df=pd.DataFrame({"left_text":test_data[0],"right_text":test_data[1],"label":test_data[2]})

    ##Get number of positiveees
    val_positives=val_df[val_df["label"]==1].shape[0]
    test_positives=test_df[test_df["label"]==1].shape[0]
    
    ## Get number of negatives
    val_negatives=val_df[val_df["label"]==0].shape[0]
    test_negatives=test_df[test_df["label"]==0].shape[0]
    
    ##Get total number of queries
    val_total=val_df.shape[0]
    test_total=test_df.shape[0]

    return val_positives, test_positives, val_negatives, test_negatives, val_total, test_total
    


###Run as script
if __name__ == "__main__":

###Replace all local paths with huggingface hub paths for easy replication as provided in the paper/website/repo
##Note that results can slightly differ as Hhyperopt is tuning the params with trials and not a fixed threshold
##All models apart from jap.

    all_models={
        "wiki-es":{"SBERT":"hiiamsid/sentence_similarity_spanish_es","LT":"dell-research-harvard/lt-wikidata-comp-es"},
    #     # "wiki-fr":{"SBERT":"dangvantuan/sentence-camembert-large","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/wiki_data/models/linkage_fr_aliases"},
    #     # "wiki-ja":{"SBERT":"oshizo/sbert-jsnli-luke-japanese-base-lite","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/wiki_data/models/linkage_ja_aliases"},
    #     # "wiki-zh":{"SBERT":"DMetaSoul/sbert-chinese-qmc-domain-v1","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/wiki_data/models/linkage_zh_aliases"},
    #     # "wiki-de":{"SBERT":"Sahajtomar/German-semantic","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/wiki_data/models/linkage_de_aliases"},
        "wiki-en":{"SBERT":"sentence-transformers/multi-qa-mpnet-base-dot-v1","LT":"dell-research-harvard/lt-wikidata-comp-en"},
        # "wiki-en-large":{"SBERT":"BAAI/bge-large-en-v1.5","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/wiki_data/models/linkage_en_aliases_large"},

    #     # "wiki_data_ja_comp_prod_industry":{"SBERT":"oshizo/sbert-jsnli-luke-japanese-base-lite","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/historicjapanese/models/lt-wikidata-comp-prod-ind-ja"},
    #     # "wiki-multi":{"SBERT":"sentence-transformers/paraphrase-multilingual-mpnet-base-v2","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/wiki_data/models/linkage_multi_aliases"},
    #     # "un-en-fine-fine":{"SBERT":"sentence-transformers/multi-qa-mpnet-base-dot-v1","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/un_data/models/linkage_un_data_en_fine_fine"},
    #     "un-en-fine-coarse":{"SBERT":"sentence-transformers/multi-qa-mpnet-base-dot-v1","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/un_data/models/linkage_un_data_en_fine_coarse"},
    #     # "un-en-fine-industry":{"SBERT":"sentence-transformers/multi-qa-mpnet-base-dot-v1","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/un_data/models/linkage_un_data_en_fine_industry"},
    #     # "un-es-fine-fine":{"SBERT":"hiiamsid/sentence_similarity_spanish_es","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/un_data/models/linkage_un_data_es_fine_fine"},
    #     # "un-es-fine-coarse":{"SBERT":"hiiamsid/sentence_similarity_spanish_es","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/un_data/models/linkage_un_data_es_fine_coarse"},
    #     # "un-es-fine-industry":{"SBERT":"hiiamsid/sentence_similarity_spanish_es","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/un_data/models/linkage_un_data_es_fine_industry"},
    #     # "un-fr-fine-fine":{"SBERT":"dangvantuan/sentence-camembert-large","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/un_data/models/linkage_un_data_fr_fine_fine"},
    #     # "un-fr-fine-coarse":{"SBERT":"dangvantuan/sentence-camembert-large","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/un_data/models/linkage_un_data_fr_fine_coarse"},
    #     # "un-fr-fine-industry":{"SBERT":"dangvantuan/sentence-camembert-large","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/un_data/models/linkage_un_data_fr_fine_industry"},
    #     # "un-multi-fine-fine":{"SBERT":"sentence-transformers/paraphrase-multilingual-mpnet-base-v2","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/un_data/models/linkage_un_data_multi_fine_fine"},
        # "un-multi-fine-coarse":{"SBERT":"sentence-transformers/paraphrase-multilingual-mpnet-base-v2","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/un_data/models/linkage_un_data_multi_fine_coarse"},
        # "un-multi-fine-industry":{"SBERT":"sentence-transformers/paraphrase-multilingual-mpnet-base-v2","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/un_data/models/linkage_un_data_multi_fine_industry"},
              }
    
    openaikey=os.environ.get("OPENAI_API_KEY")
    
    results_df_companies=make_table(all_models,openai_key=openaikey)

    results_df_companies.to_csv("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/results_df_wiki_multi_only_test_warmup01_en_coarse.csv")
    ##Reformat the table
    ## 2 decimal points
    
    results_df_companies=pd.read_csv("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/results_df_wiki_multi_only_test_warmup01_en_coarse.csv")
    results_df_companies=results_df_companies.round(2)
    results_df_companies.to_csv("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/results_df_wiki_multi_only_test_warmup01_rounded_en_coarse.csv")
    
    
    val_test_size_by_model=make_val_test_query_size_table(all_models)
    val_test_size_by_model.to_csv("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/val_test_by_model_test_en_coarse.csv")
    
    
    ###Check on historic japanese data

    model_dir="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/historicjapanese/models/"

    model_supcon="lt-historicjapanesecompanies-comp-prod-ind_supcon_full"
    model_contrastive="lt-historicjapanesecompanies-comp-prod-ind_onlinecontrastive_full"
    myopenaikey=os.environ.get("OPENAI_API_KEY")

    val_pickle_supcon=os.path.join(model_dir,model_supcon,"val_data.pickle")    
    test_pickle_supcon=os.path.join(model_dir,model_supcon,"test_data.pickle")
    val_pickle_contrastive=os.path.join(model_dir,model_contrastive,"val_data.pickle")
    test_pickle_contrastive=os.path.join(model_dir,model_contrastive,"test_data.pickle")

    
    sbert_model="oshizo/sbert-jsnli-luke-japanese-base-lite"
    lt_wiki_model=os.path.join(model_dir,"lt-wikidata-comp-prod-ind-ja")
    model_supcon=os.path.join(model_dir,model_supcon)
    model_contrastive=os.path.join(model_dir,model_contrastive)
    open_ai_model_small="text-embedding-3-small"
    open_ai_model_large="text-embedding-3-large"
    open_ai_model_ada="text-embedding-3-ada"
    
    ##Make a table of results - Edit distance, SBERT, LT ZS Wiki, LT, OpenAI
    

        # Define a function to evaluate once and return needed scores
    def get_scores(model, edit_distance, openai_key=None):
        scores = evaluate_f1_score(val_pickle_supcon, test_pickle_supcon, model, openai_key, edit_distance)
        return scores[0], scores[2]  # Return only the F1 score and threshold

    # Evaluate models
    edit_distance_score, edit_distance_threshold = get_scores(None, True)
    sbert_score, sbert_threshold = get_scores(sbert_model, False)
    lt_wiki_score, lt_wiki_threshold = get_scores(lt_wiki_model, False)
    model_supcon_score, model_supcon_threshold = get_scores(model_supcon, False)
    model_contrastive_score, model_contrastive_threshold = get_scores(model_contrastive, False)
    openai_small_score, openai_small_threshold = get_scores(open_ai_model_small, False, myopenaikey)
    openai_large_score, openai_large_threshold = get_scores(open_ai_model_large, False, myopenaikey)
    openai_ada_score, openai_ada_threshold = get_scores(open_ai_model_ada, False, myopenaikey)

    # Create the DataFrame
    results_df_japanese = pd.DataFrame({
        "edit_distance": [edit_distance_score],
        "edit_distance_threshold": [edit_distance_threshold],
        "SBERT": [sbert_score],
        "SBERT_threshold": [sbert_threshold],
        "LT ZS Wiki": [lt_wiki_score],
        "LT ZS Wiki_threshold": [lt_wiki_threshold],
        "LT_supcon": [model_supcon_score],
        "LT_supcon_threshold": [model_supcon_threshold],
        "LT_contrastive": [model_contrastive_score],
        "LT_contrastive_threshold": [model_contrastive_threshold],
        "OpenAI": [openai_small_score],
        "OpenAI_threshold": [openai_small_threshold],
        "OpenAI_large": [openai_large_score],
        "OpenAI_large_threshold": [openai_large_threshold],
        "OpenAI_ada": [openai_ada_score],
        "OpenAI_ada_threshold": [openai_ada_threshold]
    })


    ##Save the test results in the model dir as well
    save_dir=os.path.join("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/historicjapanese","historic_jp_test_results_allmethods.csv")
    results_df_japanese.to_csv(save_dir)
    
    ###Get size of japanese data
    print(get_size_of_japanese_data(val_pickle_supcon,test_pickle_supcon))
    print(get_size_of_japanese_data(val_pickle_contrastive,test_pickle_contrastive))


    # Let's run this for mexican data now
    myopenaikey=os.environ.get("OPENAI_API_KEY")

    mexican_model_dict={

                ##Multilingual models for LT_UN
                "fine-fine-multi":{"SBERT":"sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                             "LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/historicmexican/models/lt-mexicantrade4748",
                               "LT_UN":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/un_data/models/linkage_un_data_multi_fine_fine"},

    }

    # mexican_results=(make_mexican_table(mexican_model_dict,openai_key=myopenaikey,openai_model="text-embedding-3-small"))
    # mexican_results.to_csv("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/historicmexican/results_df_mexican.csv")

    val_test_size_by_model=make_val_test_query_size_table(mexican_model_dict)
    print(val_test_size_by_model)



    



