import os
import time
import warnings
from typing import List, Optional, Dict, Tuple, Iterator, Any, Union, Callable, Sequence
from linktransformer.modelling.LinkTransformer import LinkTransformer
import numpy as np
import pandas as pd
import transformers
import openai
from datasets import Dataset
from itertools import combinations
import torch
from tqdm import tqdm
import json


def _is_gemini_embedding_model(model: Any) -> bool:
    if not isinstance(model, str):
        return False
    model_name = model.lower()
    return (
        "gemini" in model_name
        or "text-embedding-004" in model_name
        or "embedding-001" in model_name
    )


def _normalize_gemini_model_name(model: str) -> str:
    if model.startswith("models/"):
        return model
    return f"models/{model}"


def infer_embeddings_with_gemini(
    strings: List[str],
    model: str,
    api_key: str,
    return_numpy: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise ImportError(
            "Gemini embeddings require `google-generativeai`. "
            "Install it with: pip install google-generativeai"
        ) from exc

    genai.configure(api_key=api_key)
    model_name = _normalize_gemini_model_name(model)

    vectors: List[List[float]] = []
    for text in strings:
        response = genai.embed_content(
            model=model_name,
            content=text,
            task_type="retrieval_document",
        )
        vector = response["embedding"] if isinstance(response, dict) else getattr(response, "embedding", None)
        if vector is None:
            raise ValueError("Gemini embedding response missing `embedding`")
        vectors.append(vector)

    if return_numpy:
        return np.array(vectors, dtype=np.float32)
    return torch.tensor(vectors, dtype=torch.float32)

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

def cosine_similarity_corresponding_pairs_torch(vector1, vector2):
    dot_product = torch.sum(vector1 * vector2, axis=1)
    norm_vector1 = torch.norm(vector1, axis=1)
    norm_vector2 = torch.norm(vector2, axis=1)
    cosine_sim = dot_product / (norm_vector1 * norm_vector2)
    return cosine_sim

def serialize_columns(df: pd.DataFrame, columns: list, sep_token: str = "</s>", model: str = None) -> list:
    """
    Serialize columns of a DataFrame into a single string.

    :param df: The DataFrame.
    :param columns: The columns to serialize.
    :param sep_token: The token to use to separate columns.
    :param model: The language model to use for tokenization (optional).
    :return: List of serialized strings.
    """
    
    def _load_sep_token(model_name: str) -> Optional[str]:
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            return tokenizer.sep_token
        except Exception:
            return None

    ###if model is string
    if isinstance(model, str):
        if model is not None:
            if "/" not in model:
                print("No base organization specified, if there is an error, it is likely because of that.")
                print(
                    f"Trying to append the model : sentence-transformers/{model} and linktransformers/{model}. Check your path otherwise!")
                ###Error handling
                loaded_sep_token = _load_sep_token(model)
                if loaded_sep_token is not None:
                    sep_token = loaded_sep_token
                else:
                    print(f"Trying sentence-transformers/{model}...")
                    loaded_sep_token = _load_sep_token("sentence-transformers/" + model)
                    if loaded_sep_token is not None:
                        sep_token = loaded_sep_token
                    else:
                        print(f"Trying linktransformers/{model}...")
                        loaded_sep_token = _load_sep_token("linktransformers/" + model)
                        if loaded_sep_token is not None:
                            sep_token = loaded_sep_token
                        else:
                            print("Could not load tokenizer for serialization. Using default sep token of </s>")
                            sep_token = "</s>"
            else:
                loaded_sep_token = _load_sep_token(model)
                if loaded_sep_token is not None:
                    sep_token = loaded_sep_token
                else:
                    print("Could not load tokenizer for serialization. Using default sep token of </s>")
                    sep_token = "</s>"
    elif isinstance(model, LinkTransformer):
        sep_token = model.tokenizer.sep_token
    else:
        sep_token = "</s>"

    if not isinstance(sep_token, str) or sep_token == "":
        sep_token = "</s>"

    return df[columns].apply(
        lambda x: sep_token.join(["" if pd.isna(v) else str(v) for v in x.tolist()]),
        axis=1,
    ).tolist()


def infer_embeddings(strings: list, model: LinkTransformer, batch_size: int = 128,
                     sbert: bool = True, openai_key: str = None, gemini_key: str = None, return_numpy=True) -> Union[np.ndarray, torch.Tensor]:
    """
    Infer embeddings for a list of strings using a language model.

    :param strings: List of strings.
    :param model: The language model to use for inference.
    :param batch_size: Batch size for inference (default: 128).
    :param sbert: If True, load model as LinkTransformer (default: True).
    :param openai_key: OpenAI API key (optional).
    :param gemini_key: Gemini API key (optional). Preferred for Gemini embedding models.
    :param return_numpy: If True, return embeddings as a numpy array (default: True). Else return a tensor.
    :return: Embeddings as a numpy array or a tensor (depending on return_numpy).
    """

    if _is_gemini_embedding_model(model):
        api_key = gemini_key or os.getenv("GEMINI_API_KEY") or openai_key
        if api_key is None:
            raise ValueError("Gemini embedding models require an API key. Pass your key via `gemini_key` or set `GEMINI_API_KEY`.")
        return infer_embeddings_with_gemini(strings, model=model, api_key=api_key, return_numpy=return_numpy)

    if openai_key is None:
        if isinstance(model, LinkTransformer):
            embeddings = model.encode(strings, batch_size=batch_size,convert_to_numpy=return_numpy,convert_to_tensor=not return_numpy)
        elif isinstance(model, str):
            if sbert:
                model = LinkTransformer(model)
            else:
                model = transformers.AutoModel.from_pretrained(model)
                print("Warning: Use of non-sentence transformer models is not recommended and is not actively supported. This will probably not work")
            embeddings = model.encode(strings, batch_size=batch_size,convert_to_numpy=return_numpy,convert_to_tensor=not return_numpy)
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
            if not openai.__version__ >= "1.0.0":
                response = openai.embeddings.create(input=split_strings[i], model=model)["data"]
                f = lambda x: x["embedding"]
            else:
                response = openai.embeddings.create(input=split_strings[i], model=model).data
                f = lambda x: x.embedding
            if return_numpy:
                embeddings.append(np.array(list(map(f, response)), dtype=np.float32))
            else: #prep as tensor
                embeddings.append(torch.tensor(list(map(f, response)), dtype=torch.float32))
        if return_numpy:
            embeddings = np.concatenate(embeddings, axis=0)
        else:
            embeddings = torch.cat(embeddings, dim=0)

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


def get_completion_from_messages(
        client: openai.OpenAI, 
        text: str,
        model: str,
        openai_key: str,
        openai_topic: str = None,
        openai_prompt: str = None,
        openai_params: Dict = None
) -> Tuple[str, int]:
    """
    This function takes a string and sends it to OpenAI API as user input and
    collects the content of the API response and the total number of tokens used

    :param client: (openai.OpenAI) OpenAI client
    :param text: (str) user input to API
    :param model: (str) name of the model to use (see "https://platform.openai.com/docs/models")
    :param openai_key: (str) OpenAI API key
    :param openai_topic: (str) topic to use for the API prompt
    :param openai_prompt: (str) prompt to use for the API
    :param openai_params: (dict) parameters to use for the API
    :returns: tuple ((str, int)): response text from API and number of tokens used
    """

    if openai_prompt is None:
        sys_prompt = f"Determine whether the text is about {openai_topic} or not. Yes/No: "
    else:
        sys_prompt = openai_prompt

    response = client.chat.completions.create(
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
    )
    return response.choices[0].message.content, response.usage.total_tokens
    

def predict_rows_with_openai(
        strings_col: List[str],
        model: str = "gpt-3.5-turbo",
        openai_key: str = None,
        openai_topic: Optional[str] = None,
        openai_prompt: Optional[str] = None,
        openai_params: Optional[dict] = None,
        label_dict: Optional[dict] = None, 
        max_retries: int = 5, 
        ratelimit_sleep_time: int = 15
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
    :param max_retries: (int) maximum number of retries if the API request fails
    :param ratelimit_sleep_time: (int) time to sleep when encountering rate-limiting errors
    :returns: (List[int]) a list of labels from OpenAI API
    """

    if openai.__version__ < "1.0.0":
        raise ValueError(f"Requires OpenAI API version 1.0.0 or higher, but you have {openai.__version__}")

    preds = []

    # create client
    client = openai.OpenAI(
        api_key=openai_key,
        timeout=openai_params["request_timeout"] if "request_timeout" in openai_params else 10
    )

    # get results from api
    for i in tqdm(range(len(strings_col))):

        # send text at row i and fetch response
        for num_retry in range(max_retries):
            try:
                r, num_tokens = get_completion_from_messages(
                    client, strings_col[i], model, openai_key, openai_topic, openai_prompt, openai_params
                )
                preds.append(r)
                break
            except Exception as e:  # handles RateLimit or Timeout Error
                tqdm.write(
                    f"Encountered error: {repr(e)}. Will retry after sleeping for {ratelimit_sleep_time * (2 ** num_retry)} seconds (attempt {num_retry + 1}/{max_retries})")
                time.sleep(ratelimit_sleep_time * (2 ** num_retry))

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
    

def apply_in_chunks(
    items: Sequence[str],
    fn: Callable[[List[str], Dict[str,Any]], List[str]],
    fn_kwargs: Optional[Dict[str,Any]] = None,
    *,
    chunk_size: int = 50,
    progress_bar: bool = True,
) -> List[str]:
    """
    Break `items` into chunks and call `fn(batch, **fn_kwargs)` on each chunk,
    returning a flat list of results in the original order.
    """
    fn_kwargs = fn_kwargs or {}
    results: List[str] = []
    chunks = [
        items[i : i + chunk_size]
        for i in range(0, len(items), chunk_size)
    ]
    iterator = tqdm(chunks, disable=not progress_bar, desc="applying transform")
    for batch in iterator:
        transformed = fn(batch, fn_kwargs)
        if len(transformed) != len(batch):
            raise ValueError("Transform function returned wrong batch size")
        results.extend(transformed)
    return results

def openai_transform(
    texts: List[str],
    params: Dict[str,Any]
) -> List[str]:
    """
    params must include:
      - client: openai.OpenAI
      - model: str
      - prompt: str
      - max_retries, ratelimit_sleep_time
      - any extra openai_params dict
    """
    client     = params["client"]
    model      = params["model"]
    prompt     = params["prompt"]
    retries    = params.get("max_retries", 5)
    backoff    = params.get("ratelimit_sleep_time", 15)
    extra      = params.get("openai_params", {})

    # append JSON instructions automatically
    json_instr = (
        "\n\nPlease return only a JSON array of strings, "
        "one entry per input, in the same order, with no extra text."
    )
    effective_prompt = prompt + json_instr

    system_msg = {"role": "system", "content": effective_prompt}
    user_msg   = {"role": "user",   "content": json.dumps(texts)}


    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[system_msg, user_msg],
                **{k: extra[k] for k in ("temperature","max_tokens","top_p",
                                         "frequency_penalty","presence_penalty")
                   if k in extra}
            )
            out = resp.choices[0].message.content
            arr = json.loads(out)
            if not isinstance(arr, list):
                raise ValueError("expected JSON list")
            if len(arr) != len(texts):
                print(arr)
                print("vs")
                print(texts)
                raise ValueError(f"expected {len(texts)} items, got {len(arr)}")
            
            return arr
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = backoff * (2 ** attempt)
            tqdm.write(f"OpenAI error {e!r}, retrying in {wait}s…")
            time.sleep(wait)
            
def transform_column(
    df: pd.DataFrame,
    column: str,
    transform_fn: Callable[[List[str], Dict[str,Any]], List[str]],
    fn_kwargs: Dict[str,Any],
    *,
    chunk_size: int = 50,
    output_column: Optional[str] = None,
    progress_bar: bool = True
) -> pd.DataFrame:
    """
    Generic: takes one column of strings, runs them through transform_fn in chunks,
    and appends a new column of the transforms.
    """
    if column not in df.columns:
        raise KeyError(column)
    ser = df[column].astype(str).tolist()
    transformed = apply_in_chunks(
        ser,
        transform_fn,
        fn_kwargs,
        chunk_size=chunk_size,
        progress_bar=progress_bar
    )
    out_col = output_column or f"{column}_transformed"
    df = df.copy()
    df[out_col] = transformed
    return df
