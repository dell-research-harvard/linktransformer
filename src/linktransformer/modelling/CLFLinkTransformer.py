from transformers import AutoModelForSequenceClassification
from typing import Optional, Union, Iterable

##Define methods for 1) Saving model card with LT details, 2)Upload to huggingface (model + tokenizer)
class CLFLinkTransformer(AutoModelForSequenceClassification):
    """Modified Sequence Classification model. """
    pass