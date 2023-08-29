import json
import pandas as pd
import itertools
import linktransformer as lt
import os



def combine_jsons(dir_path):
   ###Get all wiki dicts
    all_dicts=[]
    for filename in os.listdir(dir_path):
        if filename.endswith(".json"):
            with open(dir_path+filename) as f:
                data = json.load(f)
                all_dicts.append(data)
    
    ###Combine all dicts. If key already exists, append the values
    wiki_combined_dict={}
    for d in all_dicts:
        for k,v in d.items():
            if k in wiki_combined_dict.keys():
                wiki_combined_dict[k].extend(v)
            else:
                wiki_combined_dict[k]=v
    return wiki_combined_dict




def preprocess_wiki_aliases(wiki_combined_dict):
    ###Import the data as a dataframe. Key forms "company_id". Each value is a list of aliases - so add them as rows wuth the same company_id

    df = pd.DataFrame.from_dict(wiki_combined_dict, orient='index')
    df = df.reset_index()
    df = df.melt(id_vars=['index'], value_vars=df.columns[1:], var_name='alias_id', value_name='company_name')
    df = df.rename(columns={'index':'company_id'})
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df

def train_aliases(input_tuple):
  dir_path, model, n, save_name = input_tuple[0],input_tuple[1],input_tuple[2],input_tuple[3]
  combined_dict=combine_jsons(dir_path)
  repo_name=f"lt-wikidata-comp-multi"
  df = preprocess_wiki_aliases(combined_dict)

  ##Save the df as csv
  df.to_csv(save_name+"_train.csv")

  best_model_path = lt.train_model(
        model_path=model,
        data=df,
        clus_id_col_name=["company_id"],
        clus_text_col_names=["company_name"],
        training_args = {"num_epochs":n,"model_save_name":save_name,"save_val_test_pickles":True,
                               "wandb_names": {
                                "project": "linkage",
                                "id": "econabhishek",
                                "run": save_name,
                                "entity": "econabhishek"
                              }, 
                              "opt_model_description": f"This model was trained on a dataset consisting of company aliases from wiki data using the LinkTransformer framework. \n \
                              It was trained for {n} epochs using other defaults that can be found in the repo's LinkTransformer config file - LT_training_config.json \n  ",
                              "opt_model_lang":["en","es","fr","de","ja","zh"],
                             },
            log_wandb=True

    )




  best_model=lt.load_model(best_model_path)
  best_model.save_to_hub(repo_name = repo_name, ##Write model name here
                organization= "dell-research-harvard",
                private = None,
                commit_message = "Modified validation and training for linktransformer model",
                local_model_path = None,
                exist_ok = True,
                replace_model_card = True,
                )

  return best_model_path


###Run as script
if __name__ == "__main__":
  train_inputs = [
        ("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/wiki_aliases_copy/","sentence-transformers/paraphrase-multilingual-mpnet-base-v2",100,"linkage_multi_aliases") ]

  for input_tuple in train_inputs:
    train_aliases(input_tuple)