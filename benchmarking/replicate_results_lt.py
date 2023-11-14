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
import editdistance
from huggingface_hub import hf_hub_download



def caclulate_lev_distance(str_1,str_2):
    return editdistance.eval(str_1,str_2)


def eval_pairs_using_edit_distance(df, left_on="left_text", right_on="right_text"):
    ###If left_on or right_on len > 1, concatenate the columns

    ###now, calculate edit distance for each left_on , right on row and add a "score" column
    df["score"]=df.apply(lambda x: caclulate_lev_distance(x[left_on],x[right_on]),axis=1)

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


def calculate_retrieval_accuracy_edit(path_to_pickle):
    with open(path_to_pickle, 'rb') as handle:
        val_data = pickle.load(handle)
    ##merge the queries and corpus dataframe on the query_text and corpus_text columns using the edit distance - find edit distance between each query and corpus text
    ###The closest corpus text to each query text is the one with the lowest edit distance and should be the match

    query_df=pd.DataFrame.from_dict(val_data[0], orient='index', columns=['query_text']).reset_index()
    corpus_df=pd.DataFrame.from_dict(val_data[1], orient='index', columns=['corpus_text']).reset_index()

    print("query_df shape",query_df.shape)
    print("corpus_df shape",corpus_df.shape)
    merged_df=pd.merge(query_df,corpus_df,how="cross")

    merged_df["lev_distance"]=merged_df.apply(lambda x: caclulate_lev_distance(x["query_text"],x["corpus_text"]),axis=1)

    merged_df=merged_df.sort_values(by=["lev_distance"])

    merged_df=merged_df.drop_duplicates(subset=["index_x"],keep="first")

    merged_df=merged_df.drop_duplicates(subset=["index_y"],keep="first")


    merged_df=merged_df.reset_index()


    # # Create a new column in merged_df that indicates whether each document (cid) is relevant to its corresponding query (qid)
    def is_relevant(row):
        return 1 if row['index_y'] in val_data[2][row['index_x']] else 0

    merged_df['is_relevant'] = merged_df.apply(is_relevant, axis=1)

    # Calculate the retrieval accuracy by taking the mean of the is_relevant column
    accuracy = merged_df['is_relevant'].mean()

    return accuracy



def calculate_retrieval_accuracy_gpt(path_to_pickle,openai_key):
    with open(path_to_pickle, 'rb') as handle:
        val_data = pickle.load(handle)

    # Merge queries and corpus using LinkTransformer's lt.merge function
    merged_df = lt.merge(pd.DataFrame.from_dict(val_data[0], orient='index', columns=['query_text']).reset_index(), 
                         pd.DataFrame.from_dict(val_data[1], orient='index', columns=['corpus_text']).reset_index(), 
                         merge_type='1:m', model="text-embedding-ada-002", 
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

    table=pd.DataFrame(columns=["key","edit_distance","sbert","lt","gpt"])
    

    for key in model_dict.keys():
        sbert_model=model_dict[key]["SBERT"]
        lt_model=model_dict[key]["LT"]
        
        # hf_hub_download(repo_id=lt_model, filename="test_data.pickle")
        path_to_pickle=os.path.join(lt_model,"test_data.pickle")
        sbert_accuracy=calculate_retrieval_accuracy_lt(path_to_pickle,sbert_model)
        lt_accuracy=calculate_retrieval_accuracy_lt(path_to_pickle,lt_model)
        if openai_key is not None:
            gpt_accuracy=calculate_retrieval_accuracy_gpt(path_to_pickle,openai_key)
        else:
            gpt_accuracy=None
        editdistance_acc=calculate_retrieval_accuracy_edit(path_to_pickle)

        ##Use pd.concat to add a row to the table
        table=pd.concat([table,pd.DataFrame([[key,editdistance_acc,sbert_accuracy,lt_accuracy,gpt_accuracy]],columns=["key","edit_distance","sbert","lt","gpt"])],ignore_index=True)

    return table


def make_mexican_table(model_dict, openai_key):
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
            gpt_accuracy=calculate_retrieval_accuracy_gpt(path_to_pickle,openai_key)
        else:
            gpt_accuracy=None
        editdistance_acc=calculate_retrieval_accuracy_edit(path_to_pickle)

        ##Use pd.concat to add a row to the table
        table=pd.concat([table,pd.DataFrame([[key,editdistance_acc,sbert_accuracy,lt_un_accuracy,lt_accuracy,gpt_accuracy]],columns=["key","edit_distance","sbert","lt_un","lt","gpt"])],ignore_index=True)

    return table    

 
def evaluate_f1_score(val_pickle,test_pickle,model,openai_key=None,edit_distance=False):
    """This function needs some work - it is not complete yet"""

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

    return val_positives, test_positives
    


###Run as script
if __name__ == "__main__":

    myopenaikey= None


    all_models={
        # "wiki-es":{"SBERT":"hiiamsid/sentence_similarity_spanish_es","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/wiki_aliases_copy/models/linkage_es_aliases"},
        # "wiki-fr":{"SBERT":"dangvantuan/sentence-camembert-large","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/wiki_aliases_copy/models/linkage_fr_aliases"},
        # "wiki-ja":{"SBERT":"oshizo/sbert-jsnli-luke-japanese-base-lite","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/wiki_aliases_copy/models/linkage_ja_aliases"},
        # "wiki-zh":{"SBERT":"DMetaSoul/sbert-chinese-qmc-domain-v1","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/wiki_aliases_copy/models/linkage_zh_aliases"},
        # "wiki-de":{"SBERT":"Sahajtomar/German-semantic","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/wiki_aliases_copy/models/linkage_de_aliases"},
        # "wiki-en":{"SBERT":"sentence-transformers/multi-qa-mpnet-base-dot-v1","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/wiki_aliases_copy/models/linkage_en_aliases"},
        # "wiki_data_ja_comp_prod_industry":{"SBERT":"oshizo/sbert-jsnli-luke-japanese-base-lite","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/historicjapanese/models/lt-wikidata-comp-prod-ind-ja"},
        # "wiki-multi":{"SBERT":"sentence-transformers/paraphrase-multilingual-mpnet-base-v2","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/wiki_aliases_copy/models/linkage_multi_aliases"},
        # "un-en-fine-fine":{"SBERT":"sentence-transformers/multi-qa-mpnet-base-dot-v1","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/un_data/models/linkage_un_data_en_fine_fine"},
        # "un-en-fine-coarse":{"SBERT":"sentence-transformers/multi-qa-mpnet-base-dot-v1","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/un_data/models/linkage_un_data_en_fine_coarse"},
        # "un-en-fine-industry":{"SBERT":"sentence-transformers/multi-qa-mpnet-base-dot-v1","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/un_data/models/linkage_un_data_en_fine_industry"},
        # "un-es-fine-fine":{"SBERT":"hiiamsid/sentence_similarity_spanish_es","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/un_data/models/linkage_un_data_es_fine_fine"},
        # "un-es-fine-coarse":{"SBERT":"hiiamsid/sentence_similarity_spanish_es","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/un_data/models/linkage_un_data_es_fine_coarse"},
        # "un-es-fine-industry":{"SBERT":"hiiamsid/sentence_similarity_spanish_es","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/un_data/models/linkage_un_data_es_fine_industry"},
        # "un-fr-fine-fine":{"SBERT":"dangvantuan/sentence-camembert-large","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/un_data/models/linkage_un_data_fr_fine_fine"},
        # "un-fr-fine-coarse":{"SBERT":"dangvantuan/sentence-camembert-large","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/un_data/models/linkage_un_data_fr_fine_coarse"},
        # "un-fr-fine-industry":{"SBERT":"dangvantuan/sentence-camembert-large","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/un_data/models/linkage_un_data_fr_fine_industry"},
        # "un-multi-fine-fine":{"SBERT":"sentence-transformers/paraphrase-multilingual-mpnet-base-v2","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/un_data/models/linkage_un_data_fr_fine_fine"},
        # "un-multi-fine-coarse":{"SBERT":"sentence-transformers/paraphrase-multilingual-mpnet-base-v2","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/un_data/models/linkage_un_data_fr_fine_coarse"},
        # "un-multi-fine-industry":{"SBERT":"sentence-transformers/paraphrase-multilingual-mpnet-base-v2","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/un_data/models/linkage_un_data_fr_fine_industry"},
        "mexicantrad4748" : {"SBERT":"sentence-transformers/paraphrase-multilingual-mpnet-base-v2","LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/mexicandata/models/lt-mexicantrade4748"}
              }
    

    
    # results_df_companies=make_table(all_models,openai_key=myopenaikey)

    # results_df_companies.to_csv("results_df_wiki_multi_only.csv")
    val_test_size_by_model=make_val_test_query_size_table(all_models)
    val_test_size_by_model.to_csv("val_test_by_model.csv")
    ###Check on historic japanese data
    val_pickle="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/wiki_aliases/models/lt-historicjapanesecompanies-comp-prod-ind/val_data.pickle"
    test_pickle="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/wiki_aliases/models/lt-historicjapanesecompanies-comp-prod-ind/test_data.pickle"
    sbert_model="oshizo/sbert-jsnli-luke-japanese-base-lite"
    lt_wiki_model="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/historicjapanese/models/lt-wikidata-comp-prod-ind-ja"
    trained_lt_model = "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/wiki_aliases/models/lt-historicjapanesecompanies-comp-prod-ind"
    open_ai_model="text-embedding-ada-002"
    
    ##Make a table of results - Edit distance, SBERT, LT ZS Wiki, LT, OpenAI
    
    # evaluate_f1_score(val_pickle,test_pickle,model,openai_key=None,edit_distance=False)

    # results_df_japanese=pd.DataFrame({
    #     "edit_distance":[evaluate_f1_score(val_pickle,test_pickle,None,openai_key=None,edit_distance=True)[0]],
    #     "SBERT":[evaluate_f1_score(val_pickle,test_pickle,sbert_model,openai_key=None,edit_distance=False)[0]],
    #     "LT ZS Wiki":[evaluate_f1_score(val_pickle,test_pickle,lt_wiki_model,openai_key=None,edit_distance=False)[0]],
    #     "LT":[evaluate_f1_score(val_pickle,test_pickle,trained_lt_model,openai_key=None,edit_distance=False)[0]],
    #     "OpenAI":[evaluate_f1_score(val_pickle,test_pickle,open_ai_model,openai_key=myopenaikey,edit_distance=False)[0]],
    # })

    ###Get size of japanese data
    print(get_size_of_japanese_data(val_pickle,test_pickle))

    # results_df_japanese.to_csv("results_df_japanese.csv")

    #Let's run this for mexican data now

    # mexican_model_dict={

    #             ##Multilingual models for LT_UN
    #             "fine-fine-multi":{"SBERT":"sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    #                          "LT":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/mexicandata/models/lt-mexicantrade4748",
    #                            "LT_UN":"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/un_data/models/linkage_un_data_multi_fine_fine"},


    # }

    # mexican_results=(make_mexican_table(mexican_model_dict,openai_key=myopenaikey))
    # mexican_results.to_csv("results_df_mexican.csv")




    



