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
from transformers import CLIPProcessor, CLIPModel,AutoModel,AutoTokenizer, CLIPTokenizerFast

from PIL import Image
from torchvision import transforms

from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as T
import transformers

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
        PREPROCESS_TRANSFORM,
         MedianPad(override=(255,255,255)),
        T.ToTensor(),
        T.Resize((224, 224),antialias=True),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711),)
    ])



def collate_fn(batch):
    # Get individual fields
    input_ids = [item["input_ids"].squeeze(0) for item in batch]
    attention_masks = [item["attention_mask"].squeeze(0) for item in batch]
    pixel_values = [item["pixel_values"].squeeze(0) for item in batch]



    labels = [item["labels"] for item in batch]

    # Pad sequences
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)




    # Convert other data to tensors
    pixel_values = torch.stack(pixel_values)
    labels = torch.tensor(labels, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "pixel_values": pixel_values,
        "labels": labels
    }


class CustomDataset(Dataset):
    def __init__(self, texts, image_paths, labels, transform: T.Compose = None,tokenizer: CLIPTokenizerFast = None):
        self.texts = texts
        self.image_paths = image_paths
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image.save(f"./unproc_images_{idx}.png")
  
        text = self.texts[idx]
        
        # Process both image and text with the CLIP processor
        ###Process image
        image = CLIP_BASE_TRANSFORM(image)
        ###Process text
        processed_text = self.tokenizer(text=text, return_tensors="pt", padding=True, truncation=True)

        # ##Write transformed image  (first make it pil)
        # image_pil=transforms.ToPILImage()(image)
        # image_pil.save(f"./test_images_{idx}.png")

  

        
 

        # exit()

        return {"input_ids": processed_text.input_ids, "attention_mask": processed_text.attention_mask,"pixel_values":image, "labels": self.labels[idx]}

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


def preprocess_data(data,model,on,label_col_name,image_col_name):
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
    
    # ###Check if image column is in data
    # if image_col_name not in data.columns:
    #     raise ValueError(f"Column {image_col_name} not in data.")

    ##Ensure that image_col_name column is a string
    data[image_col_name]=data[image_col_name].astype(str)

    ##Drop images that do not exist
    data=data[data[image_col_name].apply(lambda x: os.path.exists(x))]
    ##Drop those rows for which the image path in the image_col_name column does not exist

    ###Drop if all on columns are null
    data=data.dropna(subset=on,how="all")

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
    num_labels,prop_labels = get_num_props_labels(data, label_col="label")
    print(f"Number of labels: {num_labels}")
    print(f"Proportions of labels: {prop_labels}")




    return data, num_labels, prop_labels, label_map
    

def train_mm_clf_model(data=None, model="openai/clip-vit-base-patch32", on_text=[], on_image="", label_col_name="label",
                       train_data=None, val_data=None, test_data=None, data_dir=".",
                       training_args={}, config=None,
                       eval_steps=10, save_steps=10, batch_size=None, lr=None,
                       epochs=None, model_save_dir=".", weighted_loss=True, weight_list=None,
                       wandb_log=False, wandb_name="topic", print_test_mistakes=False,freeze_encoder=False,modality="image_text",
                       modality_pooling="concat"):
    
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
    train_data, num_labels, prop_labels, label_map = preprocess_data(train_data, model, on_text, label_col_name,on_image)
    val_data, _, _, _ = preprocess_data(val_data, model, on_text, label_col_name,on_image)
    test_data, _, _, _ = preprocess_data(test_data, model, on_text, label_col_name,on_image)

    ###Save preprocessed data
    train_data.to_csv(f'{data_dir}/train.csv', encoding='utf-8', index=False)
    val_data.to_csv(f'{data_dir}/eval.csv', encoding='utf-8', index=False)
    test_data.to_csv(f'{data_dir}/test.csv', encoding='utf-8', index=False)

    
    print(len(train_data),"train")
    processor=CLIPTokenizerFast.from_pretrained(model)
    label_col_name="label"
    # Create datasets
    train_dataset = CustomDataset(train_data['text'].values, train_data[on_image].values, train_data[label_col_name].values,tokenizer=processor)
    print(train_dataset[0])
    val_dataset = CustomDataset(val_data['text'].values, val_data[on_image].values, val_data[label_col_name].values,tokenizer=processor)
    test_dataset = CustomDataset(test_data['text'].values, test_data[on_image].values, test_data[label_col_name].values,tokenizer=processor)

    print("initialized datasets")
    # Initialize model
    model = LinkTransformerClassifier(modality=modality, model_path=model, num_labels=num_labels,
                                      freeze_encoder=freeze_encoder,
                                      mm_checkpoint_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/extra_data/violence_multimodal/clip_pretrain_unlabelled_m1_newspapers_cc.pt",
                                      modality_pooling=modality_pooling)
    
    # model = LinkTransformerClassifier(modality=modality, model_path=model, num_labels=num_labels,
    #                                 freeze_encoder=freeze_encoder,
    #                                 modality_pooling=modality_pooling)
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

        
          )
    print("initialized training args")
    
    print("Training a classifier for", label_col_name)
    weight_list=weight_list if weight_list is not None else [1-prop for prop in prop_labels] if weighted_loss else None

    print(weight_list)
    class CustomTrainer(Trainer):
        def get_train_dataloader(self) -> DataLoader:            
            ##Look at the shape of the first batch
            return DataLoader(train_dataset, batch_size=batch_size*num_gpus, shuffle=True, collate_fn=collate_fn)
        def get_eval_dataloader(self, eval_dataset: Dataset) -> DataLoader:
            return DataLoader(val_dataset, batch_size=batch_size*num_gpus, collate_fn=collate_fn)
        def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
            return DataLoader(test_dataset, batch_size=batch_size*num_gpus, collate_fn=collate_fn)
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs[1]
            # compute custom loss (suppose one has 2 labels with different weights)
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(weight_list))
            ###Send to device
            loss_fct=loss_fct.to(self.args.device)
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            outputs = (loss,outputs) if return_outputs else loss
            return outputs 
    ###Custom optimiser to have different learning rate for classification head and encoder
        def create_optimizer(self):

            if isinstance(self.model, torch.nn.DataParallel):
                model = self.model.module
            else:
                model = self.model
            

            self.optimizer = torch.optim.AdamW([
                {"params": model.encoder.parameters(), "lr": lr},
                {"params": model.head.parameters(), "lr": lr*1000}
            ])

            

            return self.optimizer
        
        def create_scheduler(self, num_training_steps: int,optimizer=None):
            self.lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps
            )

            return self.lr_scheduler
    # Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=None,  # Since preprocessing is done, tokenizer isn't required here

    )

  


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
        preds = np.argmax(predictions.predictions, axis=1)
        mistakes = test_data[predictions.label_ids != preds]
        print("Mistakes in Test Set:", mistakes)

    best_model_path = trainer.state.best_model_checkpoint
    best_metric = trainer.state.best_metric

    return best_model_path,best_metric, label_map
# Usage
# results = train_mm_clf_model(df, on_text=["text1", "text2"], on_image="image_path", label_col_name="label")
