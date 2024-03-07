import pandas as pd
import pickle
import numpy as np
import linktransformer as lt


###Run as script
if __name__ == "__main__":
    ###Load the data
    path_to_japanese_data = "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/historicjapanese/ja_corrected_format.csv"

    ##open the data
    japanese_data=pd.read_csv(path_to_japanese_data, sep=',', encoding='utf-8', dtype=str)
    japanese_data=japanese_data.reset_index(drop=True)
    ###Drop any columns starting with Q
    japanese_data=japanese_data.loc[:,~japanese_data.columns.str.startswith('Q')] ##A problem with wikidata
    ##Train the link transformer model
    # Call the train_model function
    saved_model_path = lt.train_model(
        model_path="oshizo/sbert-jsnli-luke-japanese-base-lite",
        data=japanese_data,
        left_col_names=["r1","product","industry"],
        right_col_names=["r2","product","industry"],
        left_id_name=['id'],
        right_id_name=['id'],
        log_wandb=True,
        training_args={"num_epochs": 70,
                       "test_at_end": True,
                       "save_val_test_pickles": True,
                       "model_save_name": "lt-wikidata-comp-prod-ind-ja",
                       "opt_model_description": "This is a (Modern) Japanese Link Transformer model  - trained on Company <SEP> Product <SEP> Industry from wiki data.",
                       "opt_model_lang":"ja",
                       "val_perc":0.2,
                        "wandb_names": {
                                "project": "linkage",
                                "id": "econabhishek",
                                "run": "lt-wikidata-comp-prod-ind-ja",
                                "entity": "econabhishek"
                              }, 
                       
                       }
    )

    ###Save the model to hub
    best_model=lt.load_model(saved_model_path)

    best_model.save_to_hub(repo_name = "lt-wikidata-comp-prod-ind-ja", ##Write model name here
            organization= "dell-research-harvard",
            private = None,
            commit_message = "Add new LinkTransformer model.",
            local_model_path = None,
            exist_ok = True,
            replace_model_card = True,
            )


    
