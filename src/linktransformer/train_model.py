import json
import os
from typing import List, Union
from linktransformer.modified_sbert.train import train_biencoder
from linktransformer.preprocess import preprocess_any_data, preprocess_mexican_tarrif_data
from linktransformer.configs import LINKAGE_CONFIG_PATH
import pandas as pd
import pickle


def create_new_train_config(base_config_path:str=LINKAGE_CONFIG_PATH,
                            config_save_path:str="myconfig.json",
                            model_save_dir:str=None,
                            model_save_name:str=None,
                            train_batch_size:int=None,
                            num_epochs:int=None,
                            warm_up_perc:float=None,
                            learning_rate:float=None,
                            val_perc:float=None,
                            wandb_names:dict=None,
                            add_pooling_layer:bool=None,
                            language:str=None):
    """
    Function to create a training config
    :param config_save_path (str): Path to save the config
    :param base_config_path (str): Path to the base config
    :param model_save_dir (str): Path to save the model
    :param model_save_name (str): Name of the model
    :param train_batch_size (int): Batch size for training
    :param num_epochs (int): Number of epochs
    :param warm_up_perc (float): Percentage of warmup steps
    :param learning_rate (float): Learning rate
    :param val_perc (float): Percentage of validation data
    :param wandb_names (dict): Dictionary of wandb names
    :param add_pooling_layer (bool): Whether to add pooling layer
    :param language (str): Language of the model
    :return: Path to the saved config

    """
    if base_config_path is not None:
        with open(base_config_path, "r") as config_file:
            config = json.load(config_file)
    else:
        ##Load the default config
        config = "configs/linkage.json"

    ###If the user has provided a value for a parameter, update the config
    if model_save_dir is not None:
        config["model_save_dir"]=model_save_dir
    if model_save_name is not None:
        config["model_save_name"]=model_save_name
    if train_batch_size is not None:
        config["train_batch_size"]=train_batch_size
    if num_epochs is not None:
        config["num_epochs"]=num_epochs
    if warm_up_perc is not None:
        config["warm_up_perc"]=warm_up_perc
    if learning_rate is not None:
        config["learning_rate"]=learning_rate
    if val_perc is not None:
        config["val_perc"]=val_perc
    if wandb_names is not None:
        config["wandb_names"]=wandb_names
    if add_pooling_layer is not None:
        config["add_pooling_layer"]=add_pooling_layer
    if language is not None:
        config["language"]=language
    

    with open(config_save_path, "w") as config_file:
        json.dump(config, config_file)

    print("Config saved at: ", config_save_path)
    print("Keep it handy for future use")
          

    return config_save_path
    
    



def train_model(
    data: Union[str, pd.DataFrame] = None,
    train_data: Union[str, pd.DataFrame] = None,
    val_data: Union[str, pd.DataFrame] = None,
    test_data: Union[str, pd.DataFrame] = None,
    model_path: str="sentence-transformers/paraphrase-xlm-r-multilingual-v1",
    left_col_names: List[str] = ["description47"],
    right_col_names: List[str] = ['description48'],
    left_id_name: List[str] = ['tariffcode47'],
    right_id_name: List[str] = ['tariffcode48'],
    label_col_name: str = None,
    config_path: str = LINKAGE_CONFIG_PATH,
    training_args: dict = {"num_epochs":10},
    log_wandb: bool = False,
) -> str:
    """
    Train the LinkTransformer model.

    :param: model_path (str): The name of the model to use.
    :param: data (str): Path to the dataset in Excel or CSV format or a dataframe object.
    :param: left_col_names (List[str]): List of column names to use as left side data.
    :param: right_col_names (List[str]): List of column names to use as right side data.
    :param: left_id_name (List[str]): List of column names to use as identifiers for the left data.
    :param: right_id_name (List[str]): List of column names to use as identifiers for the right data,
    :param: label_col_name (str): Name of the column to use as labels. Specify this if you have data of the form (left, right, label). This type supports both positive and negative examples.
    :param: config_path (str): Path to the JSON configuration file.
    :param: training_args (dict): Dictionary of training arguments to override the config.
    :param: log_wandb (bool): Whether to log the training run on wandb.
    :return: The path to the saved best model.
    """

    # Load the configuration from the JSON file
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    ####Override config using keys that are specified
    if training_args is not None:
        for key in training_args:
            config[key]=training_args[key]
    
    ##Make model dir
    if not os.path.exists(config["model_save_dir"]):
        os.makedirs(config["model_save_dir"])
    ##Make dir for this model
    if not os.path.exists(os.path.join(config["model_save_dir"], config["model_save_name"])):
        os.makedirs(os.path.join(config["model_save_dir"], config["model_save_name"]))
        
    
    print(f"Loading config saved at {config_path}")
    print(f"Using base model: {model_path}")

    print(f"Currently wandb logging is set to {log_wandb}. Your training run can be logged on wandb by setting log_wandb=True , setting approriate parameters in the config and  making an account on wandb if you don't have one yet")

    print("Testing at end: ", config["test_at_end"])

    print("loading wiki comp data - would only work with supcon loss")
    train_data, val_data, test_data = preprocess_any_data(
        data=data,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        left_col_names=left_col_names,
        right_col_names=right_col_names,
        left_id_name=left_id_name,
        right_id_name=right_id_name,
        label_col_name=label_col_name,
        model=model_path,
        val_perc=config["val_perc"],
        large_val=config["large_val"],
        test_at_end=config["test_at_end"]
    )
    
    ##Save val and test pickles
    if config["save_val_test_pickles"]:
        print("Saving val and test pickles")
        with open(os.path.join(config["model_save_dir"], config["model_save_name"], "val_data.pickle"), "wb") as val_file:
            pickle.dump(val_data, val_file)
        with open(os.path.join(config["model_save_dir"], config["model_save_name"], "test_data.pickle"), "wb") as test_file:
            pickle.dump(test_data, test_file)

        print("Saved val and test pickles")


    ##If label_col_name is not None, specify that the eval type is classification
    if label_col_name is not None:
        config["eval_type"]="classification"
    else:
        config["eval_type"]="retrieval"

    ##Prep model directories if they don't exist
    if not os.path.exists(config["model_save_dir"]):
        os.makedirs(config["model_save_dir"])

    print("Training")
    best_model_path = train_biencoder(
        train_data=train_data,
        dev_data=val_data,
        test_data=test_data,
        base_model=model_path,
        add_pooling_layer=config["add_pooling_layer"],
        train_batch_size=config["train_batch_size"],
        num_epochs=config["num_epochs"],
        warm_up_perc=config["warm_up_perc"],
        model_save_path=os.path.join(config["model_save_dir"], config["model_save_name"]),
        wandb_names=config["wandb_names"] if log_wandb else None,
        optimizer_params={'lr': config["learning_rate"]},
        eval_type=config["eval_type"]
    )
    print(f"Best model saved on the path: {best_model_path} ")

    ##Add the dataset used in the config 
    config["training_dataset"]=data if isinstance(data,str) else "dataframe"
    ##Add the best model path in the config
    config["best_model_path"]=best_model_path
    config["base_model_path"]=model_path

    ##Save the config
    with open(os.path.join(config["model_save_dir"], config["model_save_name"], "LT_training_config.json"), "w") as config_file:
        json.dump(config, config_file)
        
    

    return best_model_path



