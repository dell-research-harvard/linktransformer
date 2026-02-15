import math
import os
import sys
from math import ceil
from typing import Dict
from unittest.mock import patch
from datasets import Dataset as HFDataset
from linktransformer.modified_sbert import losses, evaluation
import torch
import sentence_transformers
from sentence_transformers import (
    models,
    LoggingHandler,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.training_args import BatchSamplers
import logging
from transformers.utils import logging as lg
import wandb    

from linktransformer.modelling.LinkTransformer import LinkTransformer

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
#grandparentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
#sys.path.append(grandparentdir)




lg.set_verbosity_error()
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


def _build_supcon_dataset(cluster_dict):
    sentences = []
    labels = []
    cluster_label = 1
    for cluster_id in list(cluster_dict.keys()):
        for text in cluster_dict[cluster_id]:
            sentences.append(text)
            labels.append(cluster_label)
        cluster_label += 1
    print(f'{len(sentences)} training examples')
    return HFDataset.from_dict({"sentence": sentences, "label": labels})


def _build_onlinecontrastive_dataset(df):
    source_text = df['left_text'].tolist()
    target_text = df['right_text'].tolist()
    label = df['label'].tolist()
    label2int = {"same": 1, "different": 0, 1: 1, 0: 0}
    labels = [int(label2int[val]) for val in label]
    print(f'{len(labels)} training pairs')
    return HFDataset.from_dict({"sentence1": source_text, "sentence2": target_text, "label": labels})


def train_biencoder(
        train_data: dict = None,
        dev_data: dict = None,
        test_data: dict = None,
        base_model='colorfulscoop/sbert-base-ja',
        add_pooling_layer=False,
        train_batch_size=64,# The train_batch_size is an argument - you can change it for hard negatives training
        num_epochs=10,
        warm_up_perc=0.1,
        optimizer_params: Dict[str, object] = {'lr': 2e-7},
        loss_params=None,
        model_save_path="output",
        wandb_names=None,
        eval_steps_perc=0.1,
        eval_type="retrieval",
        opt_model_description=None,
        opt_model_lang=None,
        loss_type="supcon", #can be either "supcon" or "onlinecontrastive",

):

    # Logging
    if wandb_names: 
        if 'run' in wandb_names:
            wandb.init(project=wandb_names['project'], entity=wandb_names['id'], reinit=True, name=wandb_names['run'])
        else:
            wandb.init(project=wandb_names['project'], entity=wandb_names['id'], reinit=True)

        wandb.config = {
            "epochs": num_epochs,
            "batch_size": train_batch_size,
            "warm_up": warm_up_perc,
        }
        

    os.makedirs(model_save_path, exist_ok=True)

    # Base language model
    if add_pooling_layer:
        word_embedding_model = models.Transformer(base_model, max_seq_length=512)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        model = LinkTransformer(modules=[word_embedding_model, pooling_model],opt_model_description=opt_model_description,opt_model_lang=opt_model_lang)
    elif type(base_model) == LinkTransformer:
        model = base_model
    else:
        model = LinkTransformer(base_model,opt_model_description=opt_model_description,opt_model_lang=opt_model_lang)

    if wandb_names is not None:
        wandb.watch(model)


    if loss_type=="onlinecontrastive":
        train_loss = losses.OnlineContrastiveLoss_wandb(model=model,**loss_params if loss_params is not None else {},wandb_names=wandb_names)
    else:
        train_loss = losses.SupConLoss_wandb(model=model,**loss_params if loss_params is not None else {},wandb_names=wandb_names)
        


    if loss_type=="supcon":
        train_dataset = _build_supcon_dataset(train_data)
    elif loss_type=="onlinecontrastive":
        train_dataset = _build_onlinecontrastive_dataset(train_data)
    else:
        raise ValueError("loss_type can only be either 'supcon' or 'onlinecontrastive'")
    
    ##Force eval_type="classification" if loss_type=="onlinecontrastive"
    if loss_type=="onlinecontrastive":
        print("Forcing eval_type='classification' since loss_type='onlinecontrastive'")
        eval_type="classification"
    
    if eval_type=="retrieval":
        print("Evaluating on retrieval task")
        queries,corpus,relevant_docs=dev_data
        evaluators = [evaluation.InformationRetrievalEvaluator_wandb(queries,corpus,relevant_docs,wandb_names=wandb_names,name="eval")]
        seq_evaluator = sentence_transformers.evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])
        ###We also want to evaluate on the test set
        if test_data is not None:
            queries,corpus,relevant_docs=test_data
            test_evaluators = [evaluation.InformationRetrievalEvaluator_wandb(queries,corpus,relevant_docs,wandb_names=wandb_names,name="test")]
            test_evaluator = sentence_transformers.evaluation.SequentialEvaluator(test_evaluators, main_score_function=lambda scores: scores[-1])

    elif eval_type=="classification":
        print("Evaluating on pair-wise classification task")
        sentences1, sentences2, labels = dev_data
        evaluators = [evaluation.BinaryClassificationEvaluator_wandb(sentences1, sentences2, labels,wandb_names=wandb_names,name="eval")]
        seq_evaluator = sentence_transformers.evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])
        ###We also want to evaluate on the test set
        if test_data is not None:
            sentences1, sentences2, labels = test_data
            test_evaluators = [evaluation.BinaryClassificationEvaluator_wandb(sentences1, sentences2, labels,wandb_names=wandb_names,name="test")]
            test_evaluator = sentence_transformers.evaluation.SequentialEvaluator(test_evaluators, main_score_function=lambda scores: scores[-1])
    else:
        raise ValueError("eval_type can only be either 'retrieval' or 'classification'")
    


    logger.info("Evaluate model without training")
    seq_evaluator(model, epoch=0, steps=0, output_path=model_save_path)

    # Train the model using the modern trainer API
    steps_per_epoch = max(1, ceil(len(train_dataset) / max(1, train_batch_size)))
    logging_steps = max(1, ceil(steps_per_epoch * eval_steps_perc))

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed_mode = world_size > 1 or local_rank != -1

    def _build_trainer_and_train():
        training_args = SentenceTransformerTrainingArguments(
            output_dir=model_save_path,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=train_batch_size,
            learning_rate=optimizer_params.get("lr", 2e-5),
            warmup_ratio=warm_up_perc,
            logging_steps=logging_steps,
            eval_strategy="no",
            save_strategy="no",
            report_to="wandb" if wandb_names is not None else "none",
            local_rank=local_rank if distributed_mode else -1,
            ddp_backend="nccl" if distributed_mode else None,
        )

        if loss_type == "supcon":
            training_args.batch_sampler = BatchSamplers.GROUP_BY_LABEL

        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            loss=train_loss,
            evaluator=seq_evaluator,
        )
        trainer.train()
        return trainer

    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and distributed_mode:
        logger.info("Multiple CUDA devices detected with distributed launch. Using multi-GPU DDP training path.")
        trainer = _build_trainer_and_train()
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.warning(
            "Multiple CUDA devices detected without distributed launch. "
            "Skipping implicit DataParallel due to upstream incompatibility; "
            "using a single-device path. For true multi-GPU, launch with torchrun "
            "(e.g. `torchrun --nproc_per_node=<N> ...`)."
        )
        with patch("torch.cuda.device_count", return_value=1):
            trainer = _build_trainer_and_train()
    else:
        trainer = _build_trainer_and_train()

    trainer.save_model(model_save_path)

    ###Get the best model path among the saved checkpoints
    ##take the last saved checkpoints = best
    ###look in the dir and get the last checkpoint (subdirs are named as numbers, take the max one)
    # best_model_path = os.path.join(model_save_path, str(max([int(x) for x in os.listdir(model_save_path) if x.isdigit()])))
    ###Evaluate on the test set using the best model
    if test_data is not None:
        logger.info("Evaluating on the test set (0.5 of val data)")
        ##Load the best model
        print("Loading best model from:", model_save_path)
        best_model=LinkTransformer(model_save_path)
        test_evaluator(best_model, epoch=0, steps=0, output_path=model_save_path)

    return model_save_path

