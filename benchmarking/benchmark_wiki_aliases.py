import json
import pandas as pd
import itertools
import linktransformer as lt


def all_combinations(db):
    vals = []
    ids=[]
    for d in db:
        combs = list(itertools.combinations(db[d],2))
        id_rep=[d]*len(combs)
        vals = vals + combs
        ids=ids+id_rep
    df = pd.DataFrame(vals, columns=["r1", "r2"])
    df["id"]=ids
    return df

def read_json(path):
    with open(path, "r") as jsonfile:
        return json.load(jsonfile)  

def train_aliases(input_tuple):
  path, model, n, save_name = input_tuple[0],input_tuple[1],input_tuple[2],input_tuple[3]
  lang=save_name.split("_")[1]
  repo_name=f"lt-wikidata-comp-{lang}"
  df = all_combinations(read_json(path))
  ##Save the df as csv
  df.to_csv(save_name+"_train.csv")
  best_model_path=lt.train_model(
            model_path=model,
            data=df,
            left_col_names=["r1"],
            right_col_names=['r2'],
            left_id_name = ["id"],
            right_id_name = ["id"],
            training_args = {"num_epochs":n,"model_save_name":save_name,"save_val_test_pickles":True,
                               "wandb_names": {
                                "project": "linkage",
                                "id": "econabhishek",
                                "run": save_name,
                                "entity": "econabhishek"
                              }, 
                              "opt_model_description": f"This model was trained on a dataset consisting of company aliases from wiki data using the LinkTransformer framework. \n \
                              It was trained for {n} epochs using other defaults that can be found in the repo's LinkTransformer config file - LT_training_config.json \n  ",
                              "opt_model_lang":lang,
                             },
            log_wandb=True
                    )

  best_model=lt.load_model(best_model_path)
  best_model.save_to_hub(repo_name = repo_name, ##Write model name here
                organization= "dell-research-harvard",
                private = None,
                commit_message = "Add new LinkTransformer model.",
                local_model_path = None,
                exist_ok = True,
                replace_model_card = True,
                )

  return best_model_path


###Run as script
if __name__ == "__main__":
  train_inputs = [
        # ("es_aliases.json","hiiamsid/sentence_similarity_spanish_es",100,"linkage_es_aliases"),
        ("fr_aliases.json", "dangvantuan/sentence-camembert-large",100,"linkage_fr_aliases"),
      ("ja_aliases.json", "oshizo/sbert-jsnli-luke-japanese-base-lite", 100,"linkage_ja_aliases"),
      ("zh_aliases.json", "DMetaSoul/sbert-chinese-qmc-domain-v1", 100,"linkage_zh_aliases"),
      ("de_aliases.json", "T-Systems-onsite/cross-en-de-roberta-sentence-transformer", 100,"linkage_de_aliases"),
              ("en_aliases.json", "multi-qa-mpnet-base-dot-v1",100,"linkage_en_aliases" )

      ]
  all_model_paths = [train_aliases(t) for t in train_inputs]
