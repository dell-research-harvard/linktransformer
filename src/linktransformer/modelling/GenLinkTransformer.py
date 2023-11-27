from transformers import CLIPProcessor, CLIPModel,AutoModel,AutoTokenizer, CLIPVisionModel, CLIPTextModel
from typing import Optional
from torch import nn
import torch
import torch.nn.functional as F

from PIL import Image
from sentence_transformers import SentenceTransformer
import wandb
wandb.init(project="linktransformer")
import timm


# ## a general linktransformer model - text, image, multimodal
# class LinkTransformer(nn.Module):
#     def __init__(self, modality: str="text", model_path: Optional[str] = None,sentence_transformer=True,modality_pooling:str="concat"):
#         super(LinkTransformer, self).__init__()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.modality = modality
#         self.modality_pooling=modality_pooling
#         self.sentence_transformer=sentence_transformer

        
#         if modality == "text":
#             if sentence_transformer:
#                 if model_path is None:
#                     model_path = "distilroberta-base"
#                 self.encoder = SentenceTransformer(model_path)
            
#             else:
#                 if model_path is None:
#                     model_path = "distilroberta-base"
#                 self.encoder = AutoModel.from_pretrained(model_path)
#                 self.tokenizer = AutoTokenizer.from_pretrained(model_path)

#         elif modality == "image_text":
#             if model_path is None:
#                 model_path = "openai/clip-vit-base-patch32"
#             self.encoder = CLIPModel.from_pretrained(model_path)
#             self.processor = CLIPProcessor.from_pretrained(model_path)
#         else:
#             raise NotImplementedError("Modality not implemented")

    
#     def forward(self, pixel_values=None, input_ids=None, attention_mask=None, **kwargs):
#         if self.modality == "text":
#             if self.sentence_transformer:
#                 return self.encoder.forward(input_ids)["sentence_embedding"]
#             else:
#                 return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
#         elif self.modality == "image_text":
#             outputs = self.encoder(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, **kwargs)
#             if self.modality_pooling == "concat":
#                 pooled_embeds = torch.cat((outputs.text_embeds, outputs.image_embeds), dim=1)
#             elif self.modality_pooling == "mean":
#                 pooled_embeds = (outputs.text_embeds + outputs.image_embeds) / 2
#             else:
#                 raise NotImplementedError("Pooling not implemented")
#             pooled_embeds = nn.functional.normalize(pooled_embeds, p=2, dim=1)
#             return pooled_embeds
#         else:
#             raise NotImplementedError("Modality not implemented")


#     def encode(self,*inputs,**kwargs):
#         with torch.no_grad():
#             return self.forward(inputs,**kwargs)
    
#     def to_device(self, device):
#         self.device = device
#         self.to(device)

class LinkTransformer(nn.Module):
    def __init__(self, modality: str="text", model_path: Optional[str] = None,modality_pooling:str="concat",mm_checkpoint_path:Optional[str]=None):
        super(LinkTransformer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modality = modality
        self.modality_pooling=modality_pooling

        
        if modality == "text":
            ##USe only cliptext model
            if model_path is None:
                model_path = "openai/clip-vit-base-patch32" 
            self.encoder = CLIPModel.from_pretrained(model_path)
        elif modality == "image":
            ##USe only clipvision model
            # if model_path is None:
                # model_path = "openai/clip-vit-base-patch32"
            # self.encoder = CLIPModel.from_pretrained(model_path)
            self.encoder = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=2)

        elif modality == "image_text":
            if model_path is None:
                model_path = "openai/clip-vit-base-patch32"
            self.encoder = CLIPModel.from_pretrained(model_path)
            ##Check keys in encoder's state dict statting from text_model.embeddings
            model_keys=list(self.encoder.state_dict().keys())
            ##Check keys in encoder's state dict statting from text_model.embeddings
            ##If checkpoint path is provided, load the weights
            ##Check keys of mm_checkpoint_path


            if mm_checkpoint_path is not None:
                mm_checkpoint_path_keys=list(torch.load(mm_checkpoint_path).keys())

                ##Remove keys which are not in model_keys
                mm_checkpoint_path_keys=[x for x in mm_checkpoint_path_keys if x in model_keys]

                ###Remove keys which are not in mm_checkpoint_path_keys
                model_keys=[x for x in model_keys if x in mm_checkpoint_path_keys]

                ##Load the weights
                self.encoder.load_state_dict(torch.load(mm_checkpoint_path),strict=False)



        else:
            raise NotImplementedError("Modality not implemented")
        

    
    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, **kwargs):
        if self.modality == "text":
            outputs = self.encoder(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            return outputs.text_embeds
        elif self.modality == "image":
            outputs = self.encoder(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            return outputs.image_embeds

        elif self.modality == "image_text":
            outputs = self.encoder(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            if self.modality_pooling == "concat":
                pooled_embeds = torch.cat((outputs.text_embeds, outputs.image_embeds), dim=1)
            elif self.modality_pooling == "mean":
                pooled_embeds = (outputs.text_embeds + outputs.image_embeds) / 2
            else:
                raise NotImplementedError("Pooling not implemented")
            pooled_embeds = nn.functional.normalize(pooled_embeds, p=2, dim=1)
            return pooled_embeds
        else:
            raise NotImplementedError("Modality not implemented")


    def encode(self,*inputs,**kwargs):
        with torch.no_grad():
            return self.forward(inputs,**kwargs)
    
    def to_device(self, device):
        self.device = device
        self.to(device)


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_labels, dropout=0.1, num_layers=1):
        super(ClassificationHead, self).__init__()
        
        self.layers = nn.ModuleList()
        
        current_dim = input_dim
        for i in range(num_layers - 1):  # -1 to leave space for the final classifier layer
            self.layers.append(nn.Linear(current_dim, current_dim))
            self.layers.append(nn.Tanh()) ##Roberta has tanh -not sure why 
            self.layers.append(nn.Dropout(dropout))

        ###If len(layers) == 0, then the classifier is just a linear layer - so add a dropout layer
        if len(self.layers) == 0:
            self.layers.append(nn.Dropout(dropout))

        self.classifier = nn.Linear(current_dim, num_labels)
        
    def forward(self, embeddings):
        x = embeddings
        for layer in self.layers:
            x = layer(x)
        logits = self.classifier(x)
        return logits
    
    def predict(self, embeddings):
        logits = self.forward(embeddings)
        return torch.argmax(logits, dim=1)
    
    def to_device(self, device):
        self.device = device
        self.to(device)
    

class LinkTransformerClassifier(nn.Module):
    def __init__(self, modality: str = "text", model_path: Optional[str] = None, modality_pooling: str = "concat",
                 num_labels: int = 2, freeze_encoder: bool = False,mm_checkpoint_path:Optional[str]=None):
        super(LinkTransformerClassifier, self).__init__()

        # Initialize the LinkTransformer (encoder)
        self.encoder = LinkTransformer(modality=modality, model_path=model_path, 
                                        modality_pooling=modality_pooling,mm_checkpoint_path=mm_checkpoint_path)
        
        # Calculate the input dimension for the classification head
        if modality == "text":
            input_dim = self.encoder.encoder.config.projection_dim
        elif modality == "image":
            # input_dim = self.encoder.encoder.config.projection_dim
            input_dim=128
        elif modality == "image_text":
            if modality_pooling == "concat":
                input_dim = self.encoder.encoder.config.projection_dim  * 2
            else:  # "mean"
                input_dim = self.encoder.encoder.config.projection_dim 
        else:
            raise NotImplementedError("Modality not implemented")
        
        # Initialize the classification head
        self.head = ClassificationHead(input_dim=input_dim, num_labels=num_labels)

        self.num_labels = num_labels
        # Option to freeze the encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, labels=None):
        embeddings = self.encoder(pixel_values, input_ids, attention_mask)
        logits = self.head(embeddings)

        outputs = (logits,)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
            wandb.log({"train/loss":loss})

        return outputs  # (loss, logits) if labels are provided, otherwise just (logits)

    def to_device(self, device):
        self.device = device
        self.to(device)

###Run as script
if __name__ == "__main__":
    # Test text
    text_input = "Hello, my dog is cute"
    model = LinkTransformer(modality="text",sentence_transformer=False)
    output = model.encode(text_input)

    model = LinkTransformer(modality="text",sentence_transformer=True)
    output = model.encode(text_input)
    # Test image_text
    # ###Make image
    # image_input = Image.new('RGB', (224, 224), color = 'red')

    # text_input = "Hello, my dog is cute"
    # model = LinkTransformer(modality="image_text")
    # output = model.encode(image_input, text_input)
    # print(output.shape)