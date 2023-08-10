import json
import pandas as pd
import itertools
import linktransformer as lt


def all_combinations(db):
    vals = []
    for d in db:
        combs = list(itertools.combinations(db[d],2))
        vals = vals + combs
    return pd.DataFrame(vals, columns = ["r1", "r2"])

def read_json(path):
    with open(path, "r") as jsonfile:
        return json.load(jsonfile)  

def train_aliases(input_tuple):
  path, model, n, save_name = input_tuple[0],input_tuple[1],input_tuple[2],input_tuple[3]
  df = all_combinations(read_json(path))
  ##Save the df as csv
  df.to_csv(save_name+"_train.csv")
  best_model_path=lt.train_model(
            model_path=model,
            data=df,
            left_col_names=["r1"],
            right_col_names=['r2'],
            left_id_name = [],
            right_id_name = [],
            training_args = {"num_epochs":0,"model_save_name":save_name,"save_val_test_pickles":True}
                    )
  return best_model_path


###Run as script
if __name__ == "__main__":
  train_inputs = [
        # ("en_aliases.json", "multi-qa-mpnet-base-dot-v1",10,"lt/linkage_en_aliases" ),
        ("es_aliases.json","hiiamsid/sentence_similarity_spanish_es",40,"linkage_es_aliases"),
    #     ("fr_aliases.json", "dangvantuan/sentence-camembert-large",10,"lt/linkage_fr_aliases"),
    #   ("ja_aliases.json", "oshizo/sbert-jsnli-luke-japanese-base-lite", 10,"lt/linkage_ja_aliases"),
    #   ("zh_aliases.json", "DMetaSoul/sbert-chinese-qmc-domain-v1", 10,"lt/linkage_zh_aliases"),
    #   ("de_aliases.json", "T-Systems-onsite/cross-en-de-roberta-sentence-transformer", 10,"lt/linkage_de_aliases")
      ]
  all_model_paths = [train_aliases(t) for t in train_inputs]
