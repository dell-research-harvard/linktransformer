import numpy as np
import json
import pandas as pd
import os
import torch
from torch import nn
import sklearn
from datasets import load_dataset, load_metric, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, get_linear_schedule_with_warmup
import evaluate
import wandb
import sklearn.model_selection
from typing import Tuple, List, Union
from torch.utils.data import DataLoader

from transformers import Trainer, TrainingArguments, AutoConfig, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from datasets import load_metric
from transformers import set_seed
from linktransformer.utils import serialize_columns
from linktransformer.configs import CLF_CONFIG_PATH

from torch.utils.data import Dataset

from linktransformer.modelling.GenLinkTransformer import LinkTransformerClassifier
from linktransformer.modelling.ImageLinkTransformer import ImageLinkTransformer
from transformers import CLIPProcessor, CLIPModel,AutoModel,AutoTokenizer, CLIPTokenizerFast

from PIL import Image
from torchvision import transforms

from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as T
import transformers
import matplotlib.pyplot as plt


PREPROCESS_TRANSFORM = T.Compose([
    ###denoise the image slightly by blurring
    T.GaussianBlur(5, sigma=(0.1, 8.0)),
])


class MedianPad:

    def __init__(self, override=None):

        self.override = override

    def __call__(self, image):

        ##Convert to RGB 
        image = image.convert("RGB") if isinstance(image, Image.Image) else image
        image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        max_side = max(image.size)
        pad_x, pad_y = [max_side - s for s in image.size]
        # padding = (0, 0, pad_x, pad_y)
        padding = (round((10+pad_x)/2), round((5+pad_y)/2), round((10+pad_x)/2), round((5+pad_y)/2)) ##Added some extra to avoid info on the long edge

        imgarray = np.array(image)
        h, w , c= imgarray.shape
        rightb, leftb = imgarray[:,w-1,:], imgarray[:,0,:]
        topb, bottomb = imgarray[0,:,:], imgarray[h-1,:,:]
        bordervals = np.concatenate([rightb, leftb, topb, bottomb], axis=0)
        medval = tuple([int(v) for v in np.median(bordervals, axis=0)])

        return T.Pad(padding, fill=medval if self.override is None else self.override)(image)

CLIP_BASE_TRANSFORM = T.Compose([  
         MedianPad(override=(255,255,255)),
        T.ToTensor(),
        T.Resize((518, 518),antialias=True),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711),)
    ])



class ImageDataset(Dataset):
    """
    Create a PyTorch Dataset from a DataFrame containing image paths and labels.

    :param pd.DataFrame data: DataFrame containing the image paths and labels.
    :param str image_col: Name of the column containing the image paths.
    :param str label_col: Name of the column containing the labels.
    :param T.Compose transform: PyTorch transform to be applied to the images. Defaults to CLIP_BASE_TRANSFORM.

    :return: PyTorch Dataset containing the images and labels.
    """
    def __init__(self,data: pd.DataFrame, image_col: str, label_col: str, transform: T.Compose = CLIP_BASE_TRANSFORM):
        self.data = data
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform
        self.image_paths = data[image_col].tolist()
        self.labels = data[label_col].tolist()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = self.transform(image)
        label = self.labels[idx]
        return image, label
    
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return {"images": images, "labels": labels}
    



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


def preprocess_data(data,label_col_name,image_col_name):
    ##Check if on column are in data
    ##If not, error


    ###Check if label column is in data
    if label_col_name not in data.columns:
        raise ValueError(f"Column {label_col_name} not in data.")
    
    # ###Check if image column is in data
    # if image_col_name not in data.columns:
    #     raise ValueError(f"Column {image_col_name} not in data.")

    ##Ensure that image_col_name column is a string
    data[image_col_name]=data[image_col_name].astype(str)

    ##Drop images that do not exist
    data=data[data[image_col_name].apply(lambda x: os.path.exists(x))]
    ##Drop those rows for which the image path in the image_col_name column does not exist

 
    ###Drop if label column is null
    data=data.dropna(subset=[label_col_name],how="any")

    ##Check if label is an int - make int if not
    ###If string, get unique values and map to int starting from 0
    if data[label_col_name].dtype==object:
        unique_labels=data[label_col_name].unique()
        ##Sort unique labels
        unique_labels.sort()
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
    
    
    ##Rename label column to label
    data=data.rename(columns={label_col_name:"label"})
    num_labels,prop_labels = get_num_props_labels(data, label_col="label")
    print(f"Number of labels: {num_labels}")
    print(f"Proportions of labels: {prop_labels}")




    return data, num_labels, prop_labels, label_map
    
  

def train_im_clf_model(data=None, model="mobilenetv3_large_100", on_image="", label_col_name="label",
                       train_data=None, val_data=None, test_data=None, data_dir=".",
                       training_args={},
                       eval_steps=10, save_steps=10, batch_size=None, lr=None,warmup_ratio=0,
                       epochs=None, model_save_dir=".", print_test_mistakes=False):
    """
    Train a classifier model on image data.
    """
      
    # Seed setting for reproducibility
    set_seed(42)
    
    
    # Load data
    if data is not None:
        df = data if isinstance(data, pd.DataFrame) else pd.read_csv(data)
        train_data, val_data, test_data = train_test_dev_split(df, data_dir)
    
    # train_data['texts'] = train_data[on_text].agg('[SEP]'.join, axis=1)
    # val_data['texts'] = val_data[on_text].agg('[SEP]'.join, axis=1)
    # test_data['texts'] = test_data[on_text].agg('[SEP]'.join, axis=1)

    # Preprocess data
    train_data, num_labels, prop_labels, label_map = preprocess_data(train_data, label_col_name,on_image)
    val_data, _, _, _ = preprocess_data(val_data, label_col_name,on_image)
    test_data, _, _, _ = preprocess_data(test_data, label_col_name,on_image)

    ###Save preprocessed data
    train_data.to_csv(f'{data_dir}/train.csv', encoding='utf-8', index=False)
    val_data.to_csv(f'{data_dir}/eval.csv', encoding='utf-8', index=False)
    test_data.to_csv(f'{data_dir}/test.csv', encoding='utf-8', index=False)

    train_dataset=ImageDataset(train_data, image_col=on_image, label_col="label")
    val_dataset=ImageDataset(val_data, image_col=on_image, label_col="label")
    test_dataset=ImageDataset(test_data, image_col=on_image, label_col="label")

    ##Load model
    model = ImageLinkTransformer(model_name=model, pretrained=True, num_classes=num_labels)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to_device(device)

    print("initialized model")
    ##GEt number of gpus
    num_gpus=torch.cuda.device_count()
    print("Number of GPUs:",num_gpus)
    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_save_dir,
        evaluation_strategy="steps",
        save_steps=save_steps,
        eval_steps=eval_steps,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=1,
        metric_for_best_model="f1",
        prediction_loss_only=False,
        report_to="wandb",
        num_train_epochs=epochs,
        warmup_ratio=warmup_ratio

        
          )
    trainer=Trainer(model=model, args=training_args, data_collator=collate_fn,
                    train_dataset=train_dataset, 
                    eval_dataset=val_dataset,compute_metrics=compute_metrics)
    trainer.train()


    # Evaluate the model
    eval_results = trainer.evaluate()
    print("Zero shot Evaluation Results:", eval_results)

    
    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)

    ##Test
    predictions = trainer.predict(test_dataset)
    print(len(predictions.predictions))
    print("Test Results:", compute_metrics((predictions.predictions, predictions.label_ids)))
    test_results=compute_metrics((predictions.predictions, predictions.label_ids))
    ##Save results as json
    with open(f"{model_save_dir}/test_results.json","w") as f:
        json.dump(test_results,f)

    if print_test_mistakes:
        test_data=test_data.reset_index(drop=True)
        ##add preds to test data and save as csv
        preds = np.argmax(predictions.predictions, axis=1)
        test_data["preds"]=preds
        test_data.to_csv(f"{model_save_dir}/test_data_with_preds.csv")
        mistakes=test_data[test_data["label"]!=test_data["preds"]].reset_index(drop=True)
        ##Save mistakes as csv
        mistakes.to_csv(f"{model_save_dir}/mistakes.csv")
        print("Mistakes in Test Set:", mistakes)
        ##Plot mistakes and save as png
        ###Plot 25 mistakes at random
        mistakes=mistakes.head(25)
        fig = plt.figure(figsize=(20, 20), dpi=300)
        ##Plot 5 X 5 grid of images
        for i,row in mistakes.iterrows():
            ax = fig.add_subplot(5, 5, i+1)
            ax.imshow(Image.open(row[on_image]))
            ax.set_title(f"True: {row['label']}, Pred: {row['preds']}")
            ax.axis("off")
        plt.tight_layout()

        plt.savefig(f"{model_save_dir}/mistakes.png")

        
    best_model_path = trainer.state.best_model_checkpoint
    best_metric = trainer.state.best_metric

    return best_model_path,best_metric, label_map


    

    

        