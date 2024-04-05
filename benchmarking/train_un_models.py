import pandas as pd
import linktransformer as lt




###Run as script
if __name__ == "__main__":
    UN_DATA_PATH="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/un_data/un_data_all.csv"

    un_data=pd.read_csv(UN_DATA_PATH)

    ##Model 1 - Fine-fine English - The data format is one of clusters! - includes 21, 2, 11 , hs17, sitc
    lang="en"
    print(un_data["industrial"])

    un_data_subset=un_data[un_data["language"]==lang ]
    un_data_subset=un_data_subset[un_data_subset["industrial"]==0]
    print(len(un_data_subset))
    un_data_subset=un_data_subset[["CPC21code","text"]].drop_duplicates()
    print(len(un_data_subset))
    ###Remove industry ==1


    model_path="multi-qa-mpnet-base-dot-v1"
    num_epochs=100
    save_name=f"linkage_un_data_{lang}_fine_fine"
    repo_name=f"lt-un-data-fine-fine-{lang}"

    

    
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
                                "opt_model_lang":lang,
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


    # ###Model 2 - Fine-fine Spanish - The data format is one of clusters! - includes 21, 2, 11 , sitc3
    lang="es"

    un_data_subset=un_data[un_data["language"]==lang ]
    un_data_subset=un_data_subset[un_data_subset["industrial"]==0]
    print(len(un_data_subset))
    un_data_subset=un_data_subset[["CPC21code","text"]].drop_duplicates()
    print(len(un_data_subset))
    ###Remove industry ==1


    model_path="hiiamsid/sentence_similarity_spanish_es"
    num_epochs=100
    save_name=f"linkage_un_data_{lang}_fine_fine"
    repo_name=f"lt-un-data-fine-fine-{lang}"

    

    
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
                                "opt_model_lang":lang,
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
    
    #Model 3 - Fine-fine Frencch - The data format is one of clusters! - includes 21, 2, 11 , sitc3
    lang="fr"

    un_data_subset=un_data[un_data["language"]==lang ]
    un_data_subset=un_data_subset[un_data_subset["industrial"]==0]
    print(len(un_data_subset))
    un_data_subset=un_data_subset[["CPC21code","text"]].drop_duplicates()
    print(len(un_data_subset))
    ###Remove industry ==1


    model_path="dangvantuan/sentence-camembert-large"
    num_epochs=50
    save_name=f"linkage_un_data_{lang}_fine_fine"
    repo_name=f"lt-un-data-fine-fine-{lang}"

    

    
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
                                "opt_model_lang":lang,
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
    

    #Model 4  - English fine -> Industry
    lang="en"

    un_data_subset=un_data[un_data["language"]==lang ]
    un_data_subset=un_data_subset[un_data_subset["industrial"]==0]
    print(len(un_data_subset))
    un_data_subset=un_data_subset[["CPC21code","text","isic_en"]].drop_duplicates()
    print(len(un_data_subset))
    ###Remove industry ==1

    ###Remove duplicates vt text - doesn;t make sense to have the same clasification in diff industries
    un_data_subset=un_data_subset.drop_duplicates(subset=["text"])



    model_path="multi-qa-mpnet-base-dot-v1"
    num_epochs=30
    save_name=f"linkage_un_data_{lang}_fine_industry"
    repo_name=f"lt-un-data-fine-industry-{lang}"

    

    
    best_model_path = lt.train_model(
            model_path=model_path,
            data=un_data_subset,
            left_col_names=["text"],
            right_col_names=['isic_en'],
            training_args = {"num_epochs":num_epochs,"model_save_name":save_name,"save_val_test_pickles":True,
                                "wandb_names": {
                                    "project": "linkage",
                                    "id": "econabhishek",
                                    "run": save_name,
                                    "entity": "econabhishek"
                                }, 
                                "opt_model_description": f"This model was trained on a dataset prepared by linking product classifications from [UN stats](https://unstats.un.org/unsd/classifications/Econ). \n \
                                This model is designed to link different products to their industrial classification (ISIC) - trained on variation brought on by product level correspondance. It was trained for {num_epochs} epochs using other defaults that can be found in the repo's LinkTransformer config file - LT_training_config.json \n  ",
                                "opt_model_lang":lang,
                                "lr":2e-5,
                                "warmup_perc":0.5,
                                "eval_steps_perc":0.25,

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

    # Model 5  - Spanish fine -> Industry
    lang="es"

    un_data_subset=un_data[un_data["language"]==lang ]
    un_data_subset=un_data_subset[un_data_subset["industrial"]==0]
    un_data_subset=un_data_subset[["CPC21code","text","isic_es"]].drop_duplicates()
    
    ###Remove duplicates vt text - doesn;t make sense to have the same clasification in diff industries
    un_data_subset=un_data_subset.drop_duplicates(subset=["text"])
    
    model_path="hiiamsid/sentence_similarity_spanish_es"
    num_epochs=30
    save_name=f"linkage_un_data_{lang}_fine_industry"
    repo_name=f"lt-un-data-fine-industry-{lang}"

    

    
    best_model_path = lt.train_model(
            model_path=model_path,
            data=un_data_subset,
            left_col_names=["text"],
            right_col_names=['isic_es'],
            training_args = {"num_epochs":num_epochs,"model_save_name":save_name,"save_val_test_pickles":True,
                                "wandb_names": {
                                    "project": "linkage",
                                    "id": "econabhishek",
                                    "run": save_name,
                                    "entity": "econabhishek"
                                }, 
                                "opt_model_description": f"This model was trained on a dataset prepared by linking product classifications from [UN stats](https://unstats.un.org/unsd/classifications/Econ). \n \
                                This model is designed to link different products to their industrial classification (ISIC) - trained on variation brought on by product level correspondance. It was trained for {num_epochs} epochs using other defaults that can be found in the repo's LinkTransformer config file - LT_training_config.json \n  ",
                                "opt_model_lang":lang,
                                "lr":2e-5,
                                "eval_steps_perc":0.25,
                                    "warmup_perc":0.5,

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

    # Model 6  - French fine -> Industry
    lang="fr"

    un_data_subset=un_data[un_data["language"]==lang ]
    un_data_subset=un_data_subset[un_data_subset["industrial"]==0]
    print(len(un_data_subset))
    un_data_subset=un_data_subset[["CPC21code","text","isic_fr"]].drop_duplicates()
    print(len(un_data_subset))
    ###Remove industry ==1
    un_data_subset=un_data_subset.drop_duplicates(subset=["text"])


    model_path="dangvantuan/sentence-camembert-large"
    num_epochs=70
    save_name=f"linkage_un_data_{lang}_fine_industry"
    repo_name=f"lt-un-data-fine-industry-{lang}"

    un_data_subset.to_csv("un_data_subset.csv",index=False)

    # exit()
    best_model_path = lt.train_model(
            model_path=model_path,
            data=un_data_subset,
            left_col_names=["text"],
            right_col_names=['isic_fr'],
            training_args = {"num_epochs":num_epochs,"model_save_name":save_name,"save_val_test_pickles":True,
                                "wandb_names": {
                                    "project": "linkage",
                                    "id": "econabhishek",
                                    "run": save_name,
                                    "entity": "econabhishek"
                                }, 
                                "opt_model_description": f"This model was trained on a dataset prepared by linking product classifications from [UN stats](https://unstats.un.org/unsd/classifications/Econ). \n \
                                This model is designed to link different products to their industrial classification (ISIC) - trained on variation brought on by product level correspondance. It was trained for {num_epochs} epochs using other defaults that can be found in the repo's LinkTransformer config file - LT_training_config.json \n  ",
                                "opt_model_lang":lang,
                                "val_query_prop":0.5,
                                "warmup_perc":0.2,
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


# #################Start here##################################

# The above code is training three different models for linking fine-grained product classifications
# to coarse product classifications in different languages (English, Spanish, and French).
    ##Model 7  - English fine -> Coarse
    lang="en"

    un_data_subset=un_data[un_data["language"]==lang ]
    un_data_subset=un_data_subset[un_data_subset["industrial"]==0]
    print(len(un_data_subset))
    un_data_subset=un_data_subset[["CPC21code","text","en_cpc_11"]].drop_duplicates()
    print(len(un_data_subset))
    ###Remove industry ==1


    model_path= "multi-qa-mpnet-base-dot-v1" #"sentence-transformers/all-mpnet-base-v2" #  "sentence-transformers/all-mpnet-base-v2"
    num_epochs=50
    save_name=f"linkage_un_data_{lang}_fine_coarse"
    repo_name=f"lt-un-data-fine-coarse-{lang}"

    
    un_data_subset=un_data_subset.drop_duplicates(subset=["text"])

    
    best_model_path = lt.train_model(
            model_path=model_path,
            data=un_data_subset,
            left_col_names=["text"],
            right_col_names=['en_cpc_11'],
            training_args = {"num_epochs":num_epochs,"model_save_name":save_name,"save_val_test_pickles":True,
                                "wandb_names": {
                                    "project": "linkage",
                                    "id": "econabhishek",
                                    "run": save_name,
                                    "entity": "econabhishek"
                                }, 
                                "opt_model_description": f"This model was trained on a dataset prepared by linking product classifications from [UN stats](https://unstats.un.org/unsd/classifications/Econ). \n \
                                This model is designed to link different products to their coarse product classification - trained on variation brought on by product level correspondance. It was trained for {num_epochs} epochs using other defaults that can be found in the repo's LinkTransformer config file - LT_training_config.json \n  ",
                                "opt_model_lang":lang,
                                "val_query_prop":0.5,
                                "eval_steps_perc":0.25,
                                "warmup_perc":0.5,
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


    # # ##Model 8  - Spanish fine -> Coarse
    lang="es"

    un_data_subset=un_data[un_data["language"]==lang ]
    un_data_subset=un_data_subset[un_data_subset["industrial"]==0]
    print(len(un_data_subset))
    un_data_subset=un_data_subset[["CPC21code","text","es_cpc_11"]].drop_duplicates()
    print(len(un_data_subset))
    ###Remove industry ==1


    model_path= "hiiamsid/sentence_similarity_spanish_es"
    num_epochs=70
    save_name=f"linkage_un_data_{lang}_fine_coarse"
    repo_name=f"lt-un-data-fine-coarse-{lang}"

    
    un_data_subset=un_data_subset.drop_duplicates(subset=["text"])

    
    best_model_path = lt.train_model(
            model_path=model_path,
            data=un_data_subset,
            left_col_names=["text"],
            right_col_names=['es_cpc_11'],
            training_args = {"num_epochs":num_epochs,"model_save_name":save_name,"save_val_test_pickles":True,
                                "wandb_names": {
                                    "project": "linkage",
                                    "id": "econabhishek",
                                    "run": save_name,
                                    "entity": "econabhishek"
                                }, 
                                "opt_model_description": f"This model was trained on a dataset prepared by linking product classifications from [UN stats](https://unstats.un.org/unsd/classifications/Econ). \n \
                                This model is designed to link different products to their coarse product classification - trained on variation brought on by product level correspondance. It was trained for {num_epochs} epochs using other defaults that can be found in the repo's LinkTransformer config file - LT_training_config.json \n  ",
                                "opt_model_lang":lang,
                                "val_query_prop":0.5,
                                "eval_steps_perc":0.25,

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

#     # # ##Model 9  - French fine -> Coarse
    lang="fr"

    un_data_subset=un_data[un_data["language"]==lang ]
    un_data_subset=un_data_subset[un_data_subset["industrial"]==0]
    print(len(un_data_subset))
    un_data_subset=un_data_subset[["CPC21code","text","fr_cpc_11"]].drop_duplicates()
    print(len(un_data_subset))
    ###Remove industry ==1

    un_data_subset=un_data_subset.drop_duplicates(subset=["text"])

    model_path="dangvantuan/sentence-camembert-large"
    num_epochs=70
    save_name=f"linkage_un_data_{lang}_fine_coarse"
    repo_name=f"lt-un-data-fine-coarse-{lang}"

    

    
    best_model_path = lt.train_model(
            model_path=model_path,
            data=un_data_subset,
            left_col_names=["text"],
            right_col_names=['fr_cpc_11'],
            training_args = {"num_epochs":num_epochs,"model_save_name":save_name,"save_val_test_pickles":True,
                                "wandb_names": {
                                    "project": "linkage",
                                    "id": "econabhishek",
                                    "run": save_name,
                                    "entity": "econabhishek"
                                }, 
                                "opt_model_description": f"This model was trained on a dataset prepared by linking product classifications from [UN stats](https://unstats.un.org/unsd/classifications/Econ). \n \
                                This model is designed to link different products to their coarse product classification - trained on variation brought on by product level correspondance. It was trained for {num_epochs} epochs using other defaults that can be found in the repo's LinkTransformer config file - LT_training_config.json \n  ",
                                "opt_model_lang":lang,
                                "val_query_prop":0.5,
                                "eval_steps_perc":0.25,
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

