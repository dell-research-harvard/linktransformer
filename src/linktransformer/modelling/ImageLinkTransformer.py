from typing import Optional
from torch import nn
import torch
import torch.nn.functional as F

from PIL import Image
from sentence_transformers import SentenceTransformer
import wandb
wandb.init(project="linktransformer")
import timm



class ImageLinkTransformer(nn.Module):
    """ImageLinkTransformer model class - use any image model from timm"""
    def __init__(self, model_name: str,
                  pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = timm.create_model(model_name, pretrained=pretrained,num_classes=num_classes)

    def forward(self, images,labels=None):
        """If labels are provided, return loss (CE) and logits, else return logits only)"""
        logits=self.model(images)
        outputs = (logits,)
        if labels is not None:
            logits = self.model(images)
            loss = F.cross_entropy(logits, labels)
            outputs=(loss,) +  outputs
            wandb.log({"train/loss":loss})

        return outputs  # (loss), logits, (hidden_states), (attentions)

        
    def to_device(self, device):
        self.device = device
        self.to(device)
    



  





        