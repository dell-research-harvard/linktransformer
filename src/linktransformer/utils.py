import os
import time
import warnings
from typing import List, Optional

from linktransformer.modelling.LinkTransformer import LinkTransformer
import numpy as np
import pandas as pd
import transformers
import openai
from datasets import Dataset
from itertools import combinations

from tqdm import tqdm


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


def load_clf(model_path: str, num_labels: int = 2) -> transformers.AutoModelForSequenceClassification:
    """
    Load the classification model.

    :param model_path: Path to the saved model.
    :param num_labels: Number of classes to predict (default is two)
    :return: The loaded Classification model
    """
    try:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    except Exception as e:
        print(repr(e))
        raise ValueError("Error loading classification model. ")

    return model


def cosine_similarity_corresponding_pairs(vector1, vector2):
    dot_product = np.sum(vector1 * vector2, axis=1)
    norm_vector1 = np.linalg.norm(vector1, axis=1)
    norm_vector2 = np.linalg.norm(vector2, axis=1)
    cosine_sim = dot_product / (norm_vector1 * norm_vector2)
    return cosine_sim


def serialize_columns(df: pd.DataFrame, columns: list, sep_token: str = None, model: str = None) -> list:
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
            print(
                f"Trying to append the model : sentence-transformers/{model} and linktransformers/{model}. Check your path otherwise!")
            ###Error handling
            try:
                tokenizer = transformers.AutoTokenizer.from_pretrained(model)
                sep_token = tokenizer.sep_token
            except:
                try:
                    print(f"Trying sentence-transformers/{model}...")
                    tokenizer = transformers.AutoTokenizer.from_pretrained("sentence-transformers/" + model)
                    sep_token = tokenizer.sep_token
                except:
                    print(f"Trying linktransformers/{model}...")
                    tokenizer = transformers.AutoTokenizer.from_pretrained("linktransformers/" + model)
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
        char_count_string = [len(x) for x in strings]
        ##Based on character count, split the list into multiple lists - each with a max of 5000 characters
        ##This is a very rough approximation - we can do better
        ##But this is a good starting point
        ##Get the indices of the list where the split should happen. Aggregate the character count list and split when the sum is greater than 5000
        split_indices = [0]
        char_count_sum = 0
        for i in range(len(char_count_string)):
            char_count_sum += char_count_string[i]
            if char_count_sum > 5000:
                split_indices.append(i)
                char_count_sum = 0
        split_indices.append(len(char_count_string))
        ##Split the list of strings into multiple lists
        split_strings = [strings[split_indices[i]:split_indices[i + 1]] for i in range(len(split_indices) - 1)]
        ##Get the embeddings for each of the split lists
        embeddings = []
        for i in range(len(split_strings)):
            response = openai.Embedding.create(input=split_strings[i], model=model)["data"]
            f = lambda x: x["embedding"]
            embeddings.append(np.array(list(map(f, response)), dtype=np.float32))
        embeddings = np.concatenate(embeddings, axis=0)

    return embeddings


def tokenize_data_for_inference(corpus: str, name: str, hf_model: str):
    dataset = Dataset.from_dict({name: corpus})

    # Instantiate tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model)

    # Tokenize datasets
    def tokenize_function(dataset):
        return tokenizer(dataset[name], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    return tokenized_dataset


def get_completion_from_messages(text, model, openai_topic=None, openai_prompt=None, openai_params=None):
    """
    This function takes text of an article and send it to OpenAI API as user input and
    collects the content of the API response and the total number of tokens used

    :param text: (str) user input to API
    :param model: (str) name of the model to use (see "https://platform.openai.com/docs/models")
    :returns: tuple ((str, int)): response text from API and number of tokens used
    """

    if openai_prompt is None:
        sys_prompt = f"Determine whether the text is about {openai_topic} or not. Yes/No: "
    else:
        sys_prompt = openai_prompt

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": sys_prompt
            },
            {
                "role": "user",
                "content": text
            }
        ],
        temperature=openai_params["temperature"] if "temperature" in openai_params else 0,
        max_tokens=openai_params["max_tokens"] if "max_tokens" in openai_params else 1,
        top_p=openai_params["top_p"] if "top_p" in openai_params else 0,
        frequency_penalty=openai_params["frequency_penalty"] if "frequency_penalty" in openai_params else 0,
        presence_penalty=openai_params["presence_penalty"] if "presence_penalty" in openai_params else 0,
        request_timeout=openai_params["request_timeout"] if "request_timeout" in openai_params else 10,
    )
    return response.choices[0].message["content"], response.usage["total_tokens"]


def predict_rows_with_openai(
        strings_col: List[str],
        model: str = "gpt-3.5-turbo",
        openai_key: str = None,
        openai_topic: Optional[str] = None,
        openai_prompt: Optional[str] = None,
        openai_params: Optional[dict] = None,
        label_dict: Optional[dict] = None
):
    """
    This function takes a list of texts and run the texts through the API. The first part of the function
    specifies the rate-limiting parameters (e.g. how long to sleep when encountering rate-limiting errors).
    The second part retrieve the api response and save the output from API.

    :param strings_col: (List[str]) a list of texts to classify
    :param model: (str) name of the model to use (see "https://platform.openai.com/docs/models")
    :param openai_key: (str) OpenAI API key
    :param openai_topic: (str) topic to classify
    :param openai_prompt: (str) custom system prompt for OpenAI API
    :param openai_params: (dict) a dictionary to set custom parameters for OpenAI API (temperature, top_p, max_tokens, etc.)
    :param label_dict: (dict) a dictionary map text labels to numeric labels
    :returns: (List[int]) a list of labels from OpenAI API
    """

    openai.api_key = openai_key

    preds = []

    # set parameters
    if model == "gpt-3.5-turbo":
        max_tokens = 90000
        iteration_sleep_time = 0
        ratelimit_sleep_time = 60
    elif model == "gpt-4":
        max_tokens = 10000
        iteration_sleep_time = 1
        ratelimit_sleep_time = 60
    else:
        if "gpt-3.5" in model:
            max_tokens = 90000
        elif "gpt-4" in model:
            max_tokens = 10000
        else:
            max_tokens = 250000  # ada, baggage, curie, davinci (default limit)
        iteration_sleep_time = 0
        ratelimit_sleep_time = 60

    # get results from api
    accm_tokens = 0
    for i in tqdm(range(len(strings_col))):

        # sleep at when accumulated token is approaching rate-limiting threshold
        if accm_tokens > 0.8 * max_tokens:
            tqdm.write(f"Sleeping to avoid rate-limiting")
            time.sleep(ratelimit_sleep_time)
            accm_tokens = 0

        # send text at row i and fetch response
        try:
            r, num_tokens = get_completion_from_messages(
                strings_col[i], model, openai_topic, openai_prompt, openai_params
            )
            preds.append(r)

            accm_tokens += num_tokens
            time.sleep(iteration_sleep_time)
        except Exception as e:  # handles RateLimit or Timeout Error
            tqdm.write(repr(e))
            tqdm.write(f"Encountered error: will retry after sleeping")
            time.sleep(ratelimit_sleep_time)
            accm_tokens = 0

            r, num_tokens = get_completion_from_messages(
                strings_col[i], model, openai_topic, openai_prompt, openai_params
            )
            preds.append(r)

            accm_tokens += num_tokens

    if label_dict is None:
        label_dict = {"Yes": 1, "No": 0}

    try:
        preds_numeric = [label_dict[x] for x in preds]
        return preds_numeric
    except:
        warnings.warn(
            "Failed to convert OpenAI text labels to numeric labels. Text labels are kept. \
            You may want to modify the prompt or the label dict. ")
        return preds
