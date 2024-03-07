import pandas as pd
import numpy as np
import linktransformer as lt


###Run as script
if __name__ == "__main__":
    UN_DATA_PATH="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/un_data/un_data_all.csv"

    un_data=pd.read_csv(UN_DATA_PATH)
    ##Model 1 - fine - fine can be trained as is. 
    
    ##Model 1 - Fine-fine English - The data format is one of clusters! - includes 21, 2, 11 , hs17, sitc

    un_data_subset=un_data[un_data["industrial"]==0]
    print(len(un_data_subset))
    un_data_subset=un_data_subset[["CPC21code","text"]].drop_duplicates()
    print(len(un_data_subset))
    


    model_path="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    num_epochs=100
    save_name=f"linkage_un_data_multi_fine_fine"
    repo_name=f"lt-un-data-fine-fine-multi"

    

    
    best_model_path = lt.train_model(
            model_path=model_path,
            data=un_data_subset,
            clus_id_col_name=["CPC21code"],
            clus_text_col_names=["text"],
            training_args = {"num_epochs":num_epochs,"model_save_name":save_name,"save_val_test_pickles":True,
                                "wandb_names": {
                                    "project": "linkage",
                                    "id": "econabhishek",
                                    "run": save_name,
                                    "entity": "econabhishek"
                                }, 
                                "opt_model_description": f"This model was trained on a dataset prepared by linking product classifications from [UN stats](https://unstats.un.org/unsd/classifications/Econ). \n \
                                This model is designed to link different products together - trained on variation brought on by product level correspondance. It was trained for {num_epochs} epochs using other defaults that can be found in the repo's LinkTransformer config file - LT_training_config.json \n  ",
                                "opt_model_lang":["en","fr","es"],
                                "val_query_prop":0.2,
                                "batch_size": 128,
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

    ##Model 2 - Product to Industry
    ##This needs some data processing - we need to stack up language wise datasets, and add a language suffix to each of the ids
    ###That will allow us to have multiple queries for each language but common list of relevant docs while ensuring that training data 
    ###Is still based on clustering (due to graph construction). We only change the parent code to link up all the parents across
    ###Languages during training

    ###Create language wise subsets
    ##Give isic codes (self-given) by grouping by isic_en and giving group number  - this works before we have the data in the same order for all languages
    ##Then, we can use this to create a language wise subset of the data
    ##Then, we can stack them together and train the model

    #Create isic codes
    un_data=pd.read_csv(UN_DATA_PATH)

    un_data["isic_code"]=un_data.groupby("isic_en").ngroup()

    un_data_lang_subsets=[]
    for lang in ["en","es","fr"]:
        un_data_subset=un_data[un_data["language"]==lang ]
        un_data_subset=un_data_subset[un_data_subset["industrial"]==0]
        un_data_subset=un_data_subset[["CPC21code","text",f"isic_{lang}","isic_code"]].drop_duplicates()

        un_data_subset["isic_code_lang"]=un_data_subset[f"isic_code"].astype(str)+"_"+lang
        un_data_subset["isic_text"]=un_data_subset[f"isic_{lang}"].astype(str)+"_"+lang

        ##Drop isic_lang
        un_data_subset.drop(columns=[f"isic_{lang}"],inplace=True)
        un_data_lang_subsets.append(un_data_subset)



    ###now, stack them together  (by row)
    un_data_lang=pd.concat(un_data_lang_subsets,axis=0)

    ##Now, we want to train a model
    
    model_path="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    num_epochs=100
    save_name=f"linkage_un_data_multi_fine_industry"
    repo_name=f"lt-un-data-fine-industry-multi"

    

    
    best_model_path = lt.train_model(
            model_path=model_path,
            data=un_data_subset,
            left_col_names=["text"],
            right_col_names=['isic_text'],
            left_id_name=['isic_code_lang'],
            training_args = {"num_epochs":num_epochs,"model_save_name":save_name,"save_val_test_pickles":True,
                                "wandb_names": {
                                    "project": "linkage",
                                    "id": "econabhishek",
                                    "run": save_name,
                                    "entity": "econabhishek"
                                }, 
                                "opt_model_description": f"This model was trained on a dataset prepared by linking product classifications from [UN stats](https://unstats.un.org/unsd/classifications/Econ). \n \
                                This model is designed to link different products to their industrial classification (ISIC) - trained on variation brought on by product level correspondance. It was trained for {num_epochs} epochs using other defaults that can be found in the repo's LinkTransformer config file - LT_training_config.json \n  ",
                                "opt_model_lang":["en","fr","es"],
                                "val_query_prop":0.5
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

    #Model 3 - Fine product to coarse product


    #Create isic codes
    un_data=pd.read_csv(UN_DATA_PATH)


    un_data_lang_subsets=[]
    for lang in ["en","es","fr"]:
        un_data_subset=un_data[un_data["language"]==lang ]
        un_data_subset=un_data_subset[un_data_subset["industrial"]==0]
        un_data_subset=un_data_subset[["CPC21code","text",f"{lang}_cpc_11","digit3_parentcpc11"]].drop_duplicates()

        un_data_subset["digit3_parentcpc11code_lang"]=un_data_subset[f"digit3_parentcpc11"].astype(str)+"_"+lang
        un_data_subset["digit3_parentcpc11_text"]=un_data_subset[f"{lang}_cpc_11"].astype(str)+"_"+lang

        ##Drop isic_lang
        un_data_subset.drop(columns=[f"{lang}_cpc_11"],inplace=True)
        un_data_lang_subsets.append(un_data_subset)



    ###now, stack them together  (by row)
    un_data_lang=pd.concat(un_data_lang_subsets,axis=0)

    ##Now, we want to train a model
    
    model_path="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    num_epochs=100
    save_name=f"linkage_un_data_multi_fine_coarse"
    repo_name=f"lt-un-data-fine-coarse-multi"

    

    
    best_model_path = lt.train_model(
            model_path=model_path,
            data=un_data_subset,
            left_col_names=["text"],
            right_col_names=['digit3_parentcpc11_text'],
            left_id_name=['digit3_parentcpc11code_lang'],
            training_args = {"num_epochs":num_epochs,"model_save_name":save_name,"save_val_test_pickles":True,
                                "wandb_names": {
                                    "project": "linkage",
                                    "id": "econabhishek",
                                    "run": save_name,
                                    "entity": "econabhishek"
                                }, 
                                "opt_model_description": f"This model was trained on a dataset prepared by linking product classifications from [UN stats](https://unstats.un.org/unsd/classifications/Econ). \n \
                                This model is designed to link different products to their industrial classification (ISIC) - trained on variation brought on by product level correspondance. It was trained for {num_epochs} epochs using other defaults that can be found in the repo's LinkTransformer config file - LT_training_config.json \n  ",
                                "opt_model_lang":["en","fr","es"],
                                "val_query_prop":0.5
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