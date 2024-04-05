import json
import pandas as pd
import itertools
import linktransformer as lt
import os


def read_json(path):
    with open(path, "r") as jsonfile:
        return json.load(jsonfile)  
def preprocess_wiki_aliases(path_to_wiki):
    ###Import the data as a dataframe. Key forms "company_id". Each value is a list of aliases - so add them as rows wuth the same company_id
    with open(path_to_wiki) as f:
        data = json.load(f)

    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.reset_index()
    df = df.melt(id_vars=['index'], value_vars=df.columns[1:], var_name='alias_id', value_name='company_name')
    df = df.rename(columns={'index':'company_id'})
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df

def train_aliases(input_tuple):
  path, model, n, save_name = input_tuple[0],input_tuple[1],input_tuple[2],input_tuple[3]
  lang=save_name.split("_")[1]
  repo_name=f"lt-wikidata-comp-{lang}"
  df = preprocess_wiki_aliases(path)
  lang_string= lang if lang != "multi" else ['de',"en", "zh", "ja","hi", "ar", "bn", "pt", "ru", "es", "fr","ko"]


  ##Save the df as csv

  best_model_path = lt.train_model(
        model_path=model,
        data=df,
        clus_id_col_name=["company_id"],
        clus_text_col_names=["company_name"],
        training_args = {"num_epochs":n,"model_save_name":save_name,"save_val_test_pickles":True,
                         "train_batch_size": 256, 
                               "wandb_names": {
                                "project": "linkage",
                                "id": "econabhishek",
                                "run": save_name,
                                "entity": "econabhishek"
                              }, 
                              "opt_model_description": f"This model was trained on a dataset consisting of company aliases from wiki data using the LinkTransformer framework. \n \
                              It was trained for {n} epochs using other defaults that can be found in the repo's LinkTransformer config file - LT_training_config.json \n  ",
                              "opt_model_lang":lang_string,
                             },
            log_wandb=True

    )

  df.to_csv(os.path.join(best_model_path,save_name+"_train.csv"))
  print("Saved model and training data at ", best_model_path)



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
      #   ("es_aliases.json","hiiamsid/sentence_similarity_spanish_es",100,"linkage_es_aliases"),
      #   ("fr_aliases.json", "dangvantuan/sentence-camembert-large",100,"linkage_fr_aliases"),
      # ("ja_aliases.json", "oshizo/sbert-jsnli-luke-japanese-base-lite", 100,"linkage_ja_aliases"),
      # ("zh_aliases.json", "DMetaSoul/sbert-chinese-qmc-domain-v1", 100,"linkage_zh_aliases"),
      # ("de_aliases.json", "Sahajtomar/German-semantic", 100,"linkage_de_aliases"),
      #         ("en_aliases.json", "multi-qa-mpnet-base-dot-v1",100,"linkage_en_aliases" ),
      # ("multi_aliases.json","sentence-transformers/paraphrase-multilingual-mpnet-base-v2",70,"linkage_multi_aliases")
      ("en_aliases.json", "BAAI/bge-large-en-v1.5",30,"linkage_en_aliases_large"),

      ]
  all_model_paths = [train_aliases(t) for t in train_inputs]

