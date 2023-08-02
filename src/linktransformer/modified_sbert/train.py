import math
import os
import sys
from typing import Dict
from linktransformer.modified_sbert import losses, data_loaders, evaluation
from torch.utils.data import DataLoader# This is the same as torch.utils.data.DataLoader
import sentence_transformers
from sentence_transformers import models, LoggingHandler 
from sentence_transformers.datasets import SentenceLabelDataset
import logging
from transformers import logging as lg
import wandb    

from linktransformer.modified_sbert.LinkTransformer import LinkTransformer

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


def train_biencoder(
        train_data: dict = None,
        dev_data: dict = None,
        base_model='colorfulscoop/sbert-base-ja',
        add_pooling_layer=False,
        train_batch_size=64,# The train_batch_size is an argument - you can change it for hard negatives training
        num_epochs=10,
        warm_up_perc=0.1,
        optimizer_params: Dict[str, object] = {'lr': 2e-7},
        loss_params=None,
        model_save_path="output",
        wandb_names=None,
        already_clustered_train=False,
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
        model = LinkTransformer(modules=[word_embedding_model, pooling_model])
    elif type(base_model) == LinkTransformer:
        model = base_model
    else:
        model = LinkTransformer(base_model)

    if wandb_names is not None:
        wandb.watch(model)


    train_loss = losses.SupConLoss_wandb(model=model,**loss_params if loss_params is not None else {},wandb_names=wandb_names)    
        


    # Special dataset "SentenceLabelDataset" to wrap out train_set
    # It yields batches that contain at least two samples with the same label
    '''
    SentenceLabelDataset: 
    https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/datasets/SentenceLabelDataset.py

    '''

    # Load data as individuals
    ## train_data - You should organize it into cluster fformat
    train_samples = data_loaders.load_data_as_individuals(train_data, type="training",already_clustered=already_clustered_train)
    train_data_sampler = SentenceLabelDataset(train_samples)
    train_dataloader = DataLoader(train_data_sampler, batch_size=train_batch_size)

    queries,corpus,relevant_docs=dev_data
    evaluators = [evaluation.InformationRetrievalEvaluator_wandb(queries,corpus,relevant_docs,wandb_names=wandb_names)]

    seq_evaluator = sentence_transformers.evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])

    logger.info("Evaluate model without training")
    seq_evaluator(model, epoch=0, steps=0, output_path=model_save_path)

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=seq_evaluator,
        epochs=num_epochs,
        warmup_steps=math.ceil(len(train_dataloader) * num_epochs * warm_up_perc),
        output_path=model_save_path,
        evaluation_steps= math.ceil(len(train_dataloader)/10),
        checkpoint_save_steps=math.ceil(len(train_dataloader)/10),
        checkpoint_path=model_save_path,
        save_best_model=True,
        checkpoint_save_total_limit=2,
        optimizer_params = optimizer_params,
    )

    return model_save_path
