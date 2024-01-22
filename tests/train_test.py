import os
import linktransformer as lt
from linktransformer.data import DATA_DIR_PATH
import pandas as pd



def test_train_model_mexican():
    ###Test for positive pairs datasets
    # Define the path to the test dataset
    dataset_path = os.path.join(DATA_DIR_PATH, "es_mexican_products.xlsx")
    # ##Load the df
    # Call the train_model function
    saved_model_path = lt.train_model(
        model_path="hiiamsid/sentence_similarity_spanish_es",
        data=dataset_path,
        left_col_names=["description47"],
        right_col_names=['description48'],
        left_id_name=['tariffcode47'],
        right_id_name=['tariffcode48'],
        log_wandb=False,
        training_args={"num_epochs": 1,
                       "test_at_end": True,
                       "save_val_test_pickles": True,
                       "model_save_name": "check2",
                       "opt_model_description": None,
                       "opt_model_lang":None,
                       "val_perc":0.2}
    )

    model=lt.load_model(saved_model_path)

    # model.save_to_hub(repo_name = "linktransformer-models-test", ##Write model name here
    #                 organization= "dell-research-harvard",
    #                 private = None,
    #                 commit_message = "Add new LinkTransformer model.",
    #                 local_model_path = None,
    #                 exist_ok = True,
    #                 replace_model_card = True,
    #                 )

    # Add assertions to check if the training was successful and the model was saved
    assert os.path.exists(saved_model_path), "Model not saved"
    # You can add more specific assertions based on your requirements


def test_train_model_jp():
    ##Test for positive and neg pairs datasets
    dataset_path = os.path.join(DATA_DIR_PATH, "jp_pr_tk431.csv")

    ##Load the data
    df = pd.read_csv(dataset_path)

    ##Drop if tk_truth!=0 or 1
    df = df[df['tk_truth'].isin([0,1])]


   # Call the train_model function
    saved_model_path = lt.train_model(
        model_path="oshizo/sbert-jsnli-luke-japanese-base-lite",
        data=df,
        left_col_names=["source_firm_title","source_address",
                        "source_product",
                        "source_est_date",
                        "source_capital",
                        "source_bank",
                        "source_shareholder",
                        ],
        right_col_names=["tk_firm_title","tk_address",
                        "tk_product",
                        "tk_est_date",
                        "tk_capital",
                        "tk_bank",
                        "tk_shareholder"],
        label_col_name="tk_truth",
        left_id_name=['source'],
        right_id_name=['tk_path_value'],
        log_wandb=False,
        training_args={"num_epochs": 1,
                       "test_at_end": True,
                       "save_val_test_pickles": True,
                       "model_save_name": "check",
                       "opt_model_description": "test",
                       "opt_model_lang":"jp",
                       "val_perc":0.2}
    )


def test_train_model_clustering():
    #Test for cluster datasets
    dataset_path = os.path.join(DATA_DIR_PATH, "company_clusters.csv")
    ##Load the data

    saved_model_path = lt.train_model(
        model_path="sentence-transformers/all-mpnet-base-v2",
        data=dataset_path,
        clus_id_col_name=["cluster_id"],
        clus_text_col_names=["company_name"],
        log_wandb=False,
        training_args={"num_epochs": 1,
                          "test_at_end": True,
                            "save_val_test_pickles": True,
                            "model_save_name": "check2",
                            "opt_model_description": "test",
                            "opt_model_lang":"en",
                            "val_perc":0.2,
                            "batch_size": 128,
                            "large_val": True}

    )

def test_train_model_jp_onlincontrastive():
    ##Test for positive and neg pairs datasets
    dataset_path = os.path.join(DATA_DIR_PATH, "jp_pr_tk431.csv")

    ##Load the data
    df = pd.read_csv(dataset_path)

    ##Drop if tk_truth!=0 or 1
    df = df[df['tk_truth'].isin([0,1])]


   # Call the train_model function
    saved_model_path = lt.train_model(
        model_path="oshizo/sbert-jsnli-luke-japanese-base-lite",
        data=df,
        left_col_names=["source_firm_title","source_address",
                        "source_product",
                        "source_est_date",
                        "source_capital",
                        "source_bank",
                        "source_shareholder",
                        ],
        right_col_names=["tk_firm_title","tk_address",
                        "tk_product",
                        "tk_est_date",
                        "tk_capital",
                        "tk_bank",
                        "tk_shareholder"],
        label_col_name="tk_truth",
        left_id_name=['source'],
        right_id_name=['tk_path_value'],
        log_wandb=False,
        training_args={"num_epochs": 1,
                       "test_at_end": True,
                       "save_val_test_pickles": True,
                       "model_save_name": "check",
                       "opt_model_description": "test",
                       "opt_model_lang":"jp",
                       "val_perc":0.2,
                       "loss_type":"onlinecontrastive"}
    )



# if __name__ == "__main__":
    # test_train_model_jp()
    # test_train_model_mexican()
    # test_train_model_clustering()