from linktransformer.modified_sbert.LinkTransformer import LinkTransformer
import numpy as np
import pandas as pd
import transformers
import openai


def load_model(model_path: str) -> LinkTransformer:
    """
    Load a saved LinkTransformer model.

    :param model_path: Path to the saved model.
    :return: The loaded LinkTransformer model.
    """
    model = LinkTransformer(model_path)
    return model


def serialize_columns(df: pd.DataFrame, columns: list, sep_token: str, model: str = None) -> list:
    """
    Serialize columns of a DataFrame into a single string.

    :param df: The DataFrame.
    :param columns: The columns to serialize.
    :param sep_token: The token to use to separate columns.
    :param model: The language model to use for tokenization (optional).
    :return: List of serialized strings.
    """
    if model is not None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model)
        sep_token = tokenizer.sep_token

    return df[columns].apply(lambda x: sep_token.join(x.astype(str)), axis=1).tolist()


def infer_embeddings(strings: list, model: LinkTransformer, batch_size: int = 128,
                     sbert: bool = True, openai_key: str = None) -> np.ndarray:
    """
    Infer embeddings for a list of strings using a language model.

    :param strings: List of strings.
    :param model: The language model to use for inference.
    :param batch_size: Batch size for inference (default: 128).
    :param sbert: If True, load model as LinkTransformer (default: True).
    :param openai_key: OpenAI API key (optional).
    :return: Embeddings as a numpy array.
    """

    if openai_key is None:
        if isinstance(model, LinkTransformer):
            embeddings = model.encode(strings, batch_size=batch_size)
        elif isinstance(model, str):
            if sbert:
                model = LinkTransformer(model)
            else:
                model = transformers.AutoModel.from_pretrained(model)
            embeddings = model.encode(strings, batch_size=batch_size)
        else:
            raise ValueError(f"Invalid model type: {type(model)}")
    else:
        openai.api_key = openai_key
        response = openai.Embedding.create(input=strings, model=model)["data"]
        f = lambda x: x["embedding"]
        embeddings = list(map(f, response))
        embeddings = np.array(list(map(f, response)), dtype=np.float32)

    return embeddings

