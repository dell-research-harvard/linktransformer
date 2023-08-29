from linktransformer.modified_sbert.LinkTransformer import LinkTransformer
import numpy as np
import pandas as pd
import transformers
import openai
from itertools import combinations


def load_model(model_path: str) -> LinkTransformer:
    """
    Load a saved LinkTransformer model.

    :param model_path: Path to the saved model.
    :return: The loaded LinkTransformer model.
    """
    ###Throw an error if there is an error in loading the model
    try:
        model = LinkTransformer(model_path)
    except:
        print("Can't load the model, please check your path. All transformer based models from [HuggingFace](https://huggingface.co/) are supported. \
               We recommend [sentence-transformers](https://www.sbert.net/docs/pretrained_models.html) for these tasks as they are trained for \
               semantic similarity tasks. ")

    return model


def cosine_similarity_corresponding_pairs(vector1, vector2):
    dot_product = np.sum(vector1 * vector2, axis=1)
    norm_vector1 = np.linalg.norm(vector1, axis=1)
    norm_vector2 = np.linalg.norm(vector2, axis=1)
    cosine_sim = dot_product / (norm_vector1 * norm_vector2)
    return cosine_sim


    



def serialize_columns(df: pd.DataFrame, columns: list, sep_token: str=None, model: str = None) -> list:
    """
    Serialize columns of a DataFrame into a single string.

    :param df: The DataFrame.
    :param columns: The columns to serialize.
    :param sep_token: The token to use to separate columns.
    :param model: The language model to use for tokenization (optional).
    :return: List of serialized strings.
    """
    if model is not None:
        if "/" not in model:
            print("No base organization specified, if there is an error, it is likely because of that.")
            print(f"Trying to append the model : sentence-transformers/{model} and linktransformers/{model}. Check your path otherwise!" )
            ###Error handling
            try:
                tokenizer = transformers.AutoTokenizer.from_pretrained(model)
                sep_token = tokenizer.sep_token
            except:
                try:
                    print(f"Trying sentence-transformers/{model}...")
                    tokenizer = transformers.AutoTokenizer.from_pretrained("sentence-transformers/"+model)
                    sep_token = tokenizer.sep_token
                except:
                    print(f"Trying linktransformers/{model}...")
                    tokenizer = transformers.AutoTokenizer.from_pretrained("linktransformers/"+model)
                    sep_token = tokenizer.sep_token
        else:
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
        ###Open ai has a token limit on  max number of tokens per request
        char_count_string=[len(x) for x in strings]
        ##Based on character count, split the list into multiple lists - each with a max of 5000 characters
        ##This is a very rough approximation - we can do better
        ##But this is a good starting point
        ##Get the indices of the list where the split should happen. Aggregate the character count list and split when the sum is greater than 5000
        split_indices=[0]
        char_count_sum=0
        for i in range(len(char_count_string)):
            char_count_sum+=char_count_string[i]
            if char_count_sum>5000:
                split_indices.append(i)
                char_count_sum=0
        split_indices.append(len(char_count_string))
        ##Split the list of strings into multiple lists
        split_strings=[strings[split_indices[i]:split_indices[i+1]] for i in range(len(split_indices)-1)]
        ##Get the embeddings for each of the split lists
        embeddings=[]
        for i in range(len(split_strings)):
            response = openai.Embedding.create(input=split_strings[i], model=model)["data"]
            f = lambda x: x["embedding"]
            embeddings.append(np.array(list(map(f, response)), dtype=np.float32))
        embeddings=np.concatenate(embeddings,axis=0)

    return embeddings







