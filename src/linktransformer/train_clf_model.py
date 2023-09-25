
# Utils - needs edits to help tackle cases where there is no heading, byline, etc. 

import numpy as np
import json
import pandas as pd
import os
import torch
from torch import nn
import sklearn
from datasets import load_dataset, load_metric, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import wandb
import sklearn.model_selection
from typing import Tuple, List, Union

from linktransformer.utils import serialize_columns
from linktransformer.configs import CLF_CONFIG_PATH



def get_num_props_labels(pd_data: pd.DataFrame, label_col: str = "label") -> Tuple[int, List[float]]:
    """
    Get the number of unique labels and their proportions from a DataFrame.

    :param pd.DataFrame pd_data: The DataFrame containing the label data.
    :param str label_col: The name of the column containing the labels. Defaults to "label".
    
    :return: Tuple containing:
             - int: Number of unique labels.
             - List[float]: Proportions of each unique label.
    """
    unique_labels = pd_data[label_col].unique()
    label_proportions = pd_data[label_col].value_counts(normalize=True).tolist()
    return len(unique_labels), label_proportions

def train_test_dev_split(pd_data: pd.DataFrame, save_dir: str, test_perc: float = 0.15, eval_perc: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train, evaluation, and test sets and save them to CSV files.

    :param pd.DataFrame pd_data: The DataFrame to be split.
    :param str save_dir: Directory where the split CSVs should be saved.
    :param float test_perc: Proportion of data for the test set. Defaults to 0.15.
    :param float eval_perc: Proportion of data for the evaluation set. Defaults to 0.15.

    :return: Tuple containing:
             - pd.DataFrame: Train DataFrame.
             - pd.DataFrame: Evaluation DataFrame.
             - pd.DataFrame: Test DataFrame.
    """
    
    test_size = int(test_perc * len(pd_data))
    eval_size = int(eval_perc * len(pd_data))

    train_eval, test = sklearn.model_selection.train_test_split(pd_data, test_size=test_size, random_state=22)
    train, eval_data = sklearn.model_selection.train_test_split(train_eval, test_size=eval_size, random_state=17)

    # Save to CSV
    os.makedirs(save_dir, exist_ok=True)
    train.to_csv(f'{save_dir}/train.csv', encoding='utf-8', index=False)
    eval_data.to_csv(f'{save_dir}/eval.csv', encoding='utf-8', index=False)
    test.to_csv(f'{save_dir}/test.csv', encoding='utf-8', index=False)

    print(len(train), "training examples")
    print(len(eval_data), "evaluation examples")
    print(len(test), "test examples")

    return train, eval_data, test



def tokenize_data_for_finetuning(directory: str, hf_model: str, text_column: str = "text") -> 'DatasetDict':
    """
    Tokenize data for finetuning a HuggingFace model.

    :param str directory: Path to the directory containing the CSV file.
    :param str hf_model: Pretrained model identifier or path.
    :param str text_column: Name of the column containing text to be tokenized. Defaults to "text".

    :return: Tokenized dataset as a DatasetDict.
    """
    
    # Load data
    dataset = load_dataset('csv', data_files={'data': directory})

    # Instantiate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model)

    # Tokenize datasets
    def tokenize_function(dataset):
        return tokenizer(dataset[text_column], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    return tokenized_dataset["data"]

def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray],averaging_type="weighted") -> dict:
    """
    Compute evaluation metrics for predictions. Note that averaging type is always weighted for now. 

    :param Tuple[np.ndarray, np.ndarray] eval_pred: A tuple containing two arrays. The first array is logits and the second one is labels.

    :return: A dictionary with accuracy, precision, recall, and f1 score.
    """
    metric0 = evaluate.load("accuracy")
    metric1 = evaluate.load("precision")
    metric2 = evaluate.load("recall")
    metric3 = evaluate.load("f1")

    logits, labels = eval_pred

    num_classes=labels.max()+1

    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)
    
    accuracy = metric0.compute(predictions=predictions, references=labels)["accuracy"]

    if num_classes==2:
        precision = metric1.compute(predictions=predictions, references=labels)["precision"]
        recall = metric2.compute(predictions=predictions, references=labels)["recall"]
        f1 = metric3.compute(predictions=predictions, references=labels)["f1"]

    else:
        averaging_type=averaging_type
        print(metric1.compute(predictions=predictions, references=labels,average=averaging_type))

        precision = metric1.compute(predictions=predictions, references=labels,average=averaging_type)["precision"]
        recall = metric2.compute(predictions=predictions, references=labels,average=averaging_type)["recall"]
        f1 = metric3.compute(predictions=predictions, references=labels,average=averaging_type)["f1"]
    ##When using weights, the metrics become an array


    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def train_model(
        train_dataset,
        eval_dataset,
        hf_model=None,
        save_dir=None,
        num_labels=2,
        training_args={},
        wandb_log=False,
        wandb_name="topic",
        weight_list=None, 
        
):
    class BalancedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # compute custom loss (suppose one has 2 labels with different weights)
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(weight_list))
            ###Send to device
            loss_fct=loss_fct.to(self.args.device)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
    if wandb_log:
        wandb.init(project=wandb_name)
    
    ##Update training args - report_to should be none if wandb_log is false
    training_args["report_to"]="wandb" if wandb_log else "none"
    ##output_dir = save_dir if save_dir is not None else f"." 
    training_args["output_dir"]=save_dir

    model = AutoModelForSequenceClassification.from_pretrained(hf_model, num_labels=num_labels)

    training_args = TrainingArguments(
        **training_args
    )
    TrainerClass=Trainer if weight_list is None else BalancedTrainer
    # Instantiate Trainer
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    best_model_path = trainer.state.best_model_checkpoint
    best_metric = trainer.state.best_metric

    
    return best_model_path, best_metric


def evaluate_test(trained_model, predict_dataset, original_test_dir, num_labels=2,print_mistakes=False,averaging_type="weighted"):

    model = AutoModelForSequenceClassification.from_pretrained(trained_model, num_labels=num_labels)

    # Instantiate Trainer
    trainer = Trainer(model=model)

    predictions = trainer.predict(predict_dataset)

    preds = np.argmax(predictions.predictions, axis=-1)

    test_df = pd.read_csv(f"{original_test_dir}/test.csv")

    print(len(test_df), ' test examples')

    test_df["preds"] = preds

    if print_mistakes:
        # fps = test_df[(test_df["label"] == 0) & (test_df["preds"] == 1)]
        # fns = test_df[(test_df["label"] == 1) & (test_df["preds"] == 0)]
        ##WE want to generalise this to any number of labels
        fps = pd.DataFrame()
        fns = pd.DataFrame()
        for i in range(num_labels):
            fps = fps.append(test_df[(test_df["label"] == i) & (test_df["preds"] != i)])
            fns = fns.append(test_df[(test_df["label"] != i) & (test_df["preds"] == i)])

        print("Total mispredictions:", len(fps) + len(fns))
        print("False positives:", len(fps))
        print("False negatives:", len(fns))

        print("\n\n")
        print("***************** FALSE POSITIVES *****************")
        for i in list(fps.index.values):
            print(fps["text"][i])
            print("*****")

        print("\n\n")
        print("***************** FALSE NEGATIVES *****************")
        for i in list(fns.index.values):
            print(fns["text"][i])
            print("*****")

    print("***Test results***")
    metric0 = evaluate.load("accuracy")
    metric2 = evaluate.load("precision")
    metric1 = evaluate.load("recall")
    metric3 = evaluate.load("f1")


    results_dict={}
    results_dict["test/accuracy"]=metric0.compute(predictions=preds, references=predictions.label_ids)["accuracy"]

    if num_labels==2:
        averaging_type=None
        results_dict["test/precision"]=metric2.compute(predictions=preds, references=predictions.label_ids)["precision"]
        results_dict["test/recall"]=metric1.compute(predictions=preds, references=predictions.label_ids)["recall"]
        results_dict["test/f1"]=metric3.compute(predictions=preds, references=predictions.label_ids)["f1"]

    else:
        averaging_type=averaging_type
        results_dict["test/precision"]=metric2.compute(predictions=preds, references=predictions.label_ids,average=averaging_type)["precision"]
        results_dict["test/recall"]=metric1.compute(predictions=preds, references=predictions.label_ids,average=averaging_type)["recall"]
        results_dict["test/f1"]=metric3.compute(predictions=preds, references=predictions.label_ids,average=averaging_type)["f1"]
    
    print(results_dict)
    return results_dict


def save_training_splits(save_dir,train_data,val_data,test_data):
    os.makedirs(save_dir, exist_ok=True)
    train_data.to_csv(f'{save_dir}/train.csv', encoding='utf-8', index=False)
    val_data.to_csv(f'{save_dir}/eval.csv', encoding='utf-8', index=False)
    test_data.to_csv(f'{save_dir}/test.csv', encoding='utf-8', index=False)

    print(len(train_data), "training examples")
    print(len(val_data), "dev examples")
    print(len(test_data), "test examples")

def preprocess_data(data,model,on,label_col_name):
    ##Check if on column are in data
    ##If not, error
    if isinstance(on,str):
        on=[on]

    for col_name in on:
        if col_name not in data.columns:
            raise ValueError(f"Column {col_name} not in data.")
        
    ###Check if label column is in data
    if label_col_name not in data.columns:
        raise ValueError(f"Column {label_col_name} not in data.")
    
        
    ###Drop if all on columns are null
    data=data.dropna(subset=on,how="all")

    ###Drop if label column is null
    data=data.dropna(subset=[label_col_name],how="any")

    ##Check if label is an int - make int if not
    ###If string, get unique values and map to int starting from 0
    if data[label_col_name].dtype==object:
        unique_labels=data[label_col_name].unique()
        label_map={label:i for i,label in enumerate(unique_labels)}
        data[label_col_name]=data[label_col_name].map(label_map)
        ##Print mapping
        print(f"Label mapping: {label_map}")
    ###If float, make int
    elif data[label_col_name].dtype==float:
        data[label_col_name]=data[label_col_name].astype(int)
        label_map={i:i for i in range(data[label_col_name].max()+1)}
    ###If int, do nothing
    elif data[label_col_name].dtype==int:
        label_map={i:i for i in range(data[label_col_name].max()+1)}
    else:
        raise ValueError(f"Label column {label_col_name} is not an int, float or string.")

    ##Check if on columns are strings - make strings if not
    for col_name in on:
        if data[col_name].dtype!=object:
            data[col_name]=data[col_name].astype(str)


    ##Serialize columns using the model's tokenizer
    if len(on)> 1:
        data["text"]=serialize_columns(data,on,model)
    else:
        data["text"]=data[on[0]]
    
    ##Rename label column to label
    data=data.rename(columns={label_col_name:"label"})
    num_labels,prop_labels = get_num_props_labels(data)
    print(f"Number of labels: {num_labels}")
    print(f"Proportions of labels: {prop_labels}")




    return data, num_labels, prop_labels, label_map




def load_default_training_args(config_path):
    with open(config_path, "r") as f:
        return json.load(f)



def train_clf_model(data=None,model="distilroberta-base",on=[],label_col_name="label",train_data=None,val_data=None,test_data=None,data_dir=".",
                    training_args={},config=CLF_CONFIG_PATH,
                    eval_steps=None,save_steps=None,batch_size=None,lr=None,
                    epochs=None,model_save_dir=".", weighted_loss=False,weight_list=None,
                    wandb_log=False,wandb_name="topic",
                    print_test_mistakes=False):
    """
    Trains a text classification model using Hugging Face's Transformers library.
    
    :param data: (str/DataFrame, optional) Path to the CSV file or a DataFrame object containing the training data.
    :param model: (str, default="distilroberta-base") The name of the Hugging Face model to be used.
    :param on: (list, default=[]) List of column names that are used as input features.
    :param label_col_name: (str, default="label") The column name in the data that contains the labels.
    :param train_data: (str/DataFrame, optional) Training dataset if `data` is not provided.
    :param val_data: (str/DataFrame, optional) Validation dataset if `data` is not provided.
    :param test_data: (str/DataFrame, optional) Test dataset if `data` is not provided.
    :param data_dir: (str, default=".") Directory where training data splits are saved.
    :param training_args: (dict, default={}) Training arguments for the Hugging Face Trainer.
    :param config: (str, default=CLF_CONFIG_PATH) Path to the default config file.
    :param eval_steps: (int, optional) Evaluation interval in terms of steps.
    :param save_steps: (int, optional) Model saving interval in terms of steps.
    :param batch_size: (int, optional) Batch size for training and evaluation.
    :param lr: (float, optional) Learning rate.
    :param epochs: (int, optional) Number of training epochs.
    :param model_save_dir: (str, default=".") Directory where the trained model will be saved.
    :param weighted_loss: (bool, default=False) If true, uses weighted loss based on class frequencies.
    :param weight_list: (list, optional) Weights for each class in the loss function.
    :param wandb_log: (bool, default=False) If true, logs metrics to Weights & Biases.
    :param wandb_name: (str, default="topic") Name of the Weights & Biases project.
    :param print_test_mistakes: (bool, default=False) If true, prints the misclassified samples in the test dataset.
    
    :return: 
        - best_model_path (str): Path to the directory of the best saved model.
        - best_metric (float): The best metric value achieved during training.
        - label_map (dict): Mapping of labels to their respective integer values.
        
    Note:
        Either the `data` parameter or all of `train_data`, `val_data`, and `test_data` should be provided. If only
        `data` is provided, it will be split into train, validation, and test sets.
    """
    
    if train_data is None and val_data is None and test_data is None:
        ##Read csv or dataframe object
        data = pd.read_csv(data) if isinstance(data,str) else data
        train_data,val_data,test_data = train_test_dev_split(data, data_dir, test_perc=0.15, eval_perc=0.15)
        train_data,num_labels, prop_labels, label_map=preprocess_data(train_data,model,on,label_col_name)
        val_data,_,_,_=preprocess_data(val_data,model,on,label_col_name)
        test_data,_,_,_=preprocess_data(test_data,model,on,label_col_name)
        save_training_splits(data_dir,train_data,val_data,test_data)

    else:
        data=None
        train_data = pd.read_csv(train_data) if isinstance(train_data,str) else train_data
        val_data = pd.read_csv(val_data) if isinstance(val_data,str) else val_data
        test_data = pd.read_csv(test_data) if isinstance(test_data,str) else test_data
        train_data,num_labels, prop_labels, label_map=preprocess_data(train_data,model,on,label_col_name)
        val_data,_,_,_=preprocess_data(val_data,model,on,label_col_name)
        test_data,_,_,_=preprocess_data(test_data,model,on,label_col_name)
        save_training_splits(data_dir,train_data,val_data,test_data)
        
    ##Tokenize data
    datasets = {}
    for dataset in ["train", "eval", "test"]:
        datasets[dataset] = tokenize_data_for_finetuning(
            directory=os.path.join(data_dir,f'{dataset}.csv'),
            hf_model=model
        )

    specified_training_args = training_args

    ##Load config as training args
    training_args = load_default_training_args(config)
    
    ##Override if training_args is not empty
    if len(training_args)>0:
        training_args.update(specified_training_args)
    print("Updating training args with: ", specified_training_args)
    
    ##Override some args
    if lr is not None:
        training_args["learning_rate"] = lr
    if batch_size is not None:
        training_args["per_device_train_batch_size"] = batch_size
        training_args["per_device_eval_batch_size"] = batch_size
    if epochs is not None:
        training_args["num_train_epochs"] = epochs
    if eval_steps is not None:
        training_args["evaluation_strategy"] = "steps"
        training_args["eval_steps"] = eval_steps
    if save_steps is not None:
        training_args["save_steps"] = save_steps

    
    

    ##Get weight list from data if needed
    weight_list=weight_list if weight_list is not None else [1-prop for prop in prop_labels] if weighted_loss else None

    ##Train model
    best_model_path, best_metric = train_model(
        datasets["train"],
        datasets["eval"],
        model,
        model_save_dir,
        num_labels=num_labels,
        training_args=training_args,
        wandb_log=wandb_log,
        wandb_name=wandb_name,
        weight_list=weight_list, 
        )
    
    print("Note the path to the best model: ", best_model_path)




    ###Save label map 
    if best_model_path is  None:
        print("No model was saved because the trained model is not better than the pretraind one. Try tuning the hyperparameters or training for longer")
    else:    
        ##Evaluate model (test)
        test_results=evaluate_test(best_model_path,datasets["test"],data_dir,num_labels=num_labels,print_mistakes=print_test_mistakes)

        ##Log test results to wandb
        if wandb_log:
            wandb.log(test_results)
        with open(f"{best_model_path}/label_map.json","w") as f:
            json.dump(label_map,f)
        ##Saving tokenizer to the same directory as the model
    
        if not os.path.exists(f"{best_model_path}/tokenizer"):
            tokenizer = AutoTokenizer.from_pretrained(model)
            tokenizer.save_pretrained(best_model_path)

    
    return best_model_path, best_metric, label_map



