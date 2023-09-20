import pandas as pd
import pickle
import numpy as np
import linktransformer as lt
import os



###run as script

if __name__ == "__main__":
     ###Load the data

   # Call the train_model function
    # Define the path to the test dataset
    dataset_path = os.path.join(lt.DATA_DIR_PATH, "es_mexican_products.xlsx")

    ##Do some light preprocessing (it is all caps right now)
    df=pd.read_excel(dataset_path)
    df=df.copy()
    ##Drop missing descriptions
    df=df.dropna(subset=["description47","description48"])
    ##Make descriptions lowercase
    df=df.applymap(lambda x: x.lower() if isinstance(x,str) else x)
    ##Drop dropna rows
    df=df.dropna()
    
    # Call the train_model function
    saved_model_path = lt.train_model(
        model_path="hiiamsid/sentence_similarity_spanish_es",
        data=df,
        left_col_names=["description47"],
        right_col_names=['description48'],
        left_id_name=['tariffcode47'],
        right_id_name=['tariffcode48'],
        log_wandb=False,
        training_args={"num_epochs": 100,
                       "test_at_end": True,
                       "save_val_test_pickles": True,
                       "model_save_name": "lt-mexicantrade4748_test",
                       "opt_model_description": None,
                       "opt_model_lang":None,
                       "val_perc":0.2,
                       "wandb_names":{
                                     "id": "econabhishek",
                                    "run": "lt-mexicantrade4748",
                                    "project": "linkage",
                                    "entity": "econabhishek" }}
    )
 



    # # ###Save the model to hub
    # best_model=lt.load_model(saved_model_path)

    # best_model.save_to_hub(repo_name = "lt-mexicantrade4748", ##Write model name here
    #         organization= "dell-research-harvard",
    #         private = None,
    #         commit_message = "Add new LinkTransformer model.",
    #         local_model_path = None,
    #         exist_ok = True,
    #         replace_model_card = True,
    #         )


    
