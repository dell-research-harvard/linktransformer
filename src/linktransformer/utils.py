import importlib
import json
import logging
import os
import time
import warnings
from typing import List, Optional, Dict, Tuple, Iterator, Any, Union, Callable, Sequence
from linktransformer.modelling.LinkTransformer import LinkTransformer
import numpy as np
import pandas as pd
import transformers
import torch
from packaging.version import Version
from tqdm import tqdm

logger = logging.getLogger(__name__)

# pip distribution name and extra to suggest when an optional module is missing
_OPTIONAL_MODULES = {
    "openai": ("openai", None),
    "datasets": ("datasets", "train"),
    "google.genai": ("google-genai", "gemini"),
    "hdbscan": ("hdbscan", None),
    "wandb": ("wandb", "train"),
}


def _require(module_name: str, min_version: Optional[str] = None):
    """
    Import an optional dependency on first use, with an actionable error if it is missing.

    :param module_name: Importable module name, e.g. "openai" or "google.genai".
    :param min_version: Minimum supported version (compared against ``module.__version__``).
    :return: The imported module.
    """
    dist_name, extra = _OPTIONAL_MODULES.get(module_name, (module_name, None))
    hint = f"pip install 'linktransformer[{extra}]'" if extra else f"pip install {dist_name}"
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"This feature requires `{dist_name}`. Install it with: {hint}") from exc
    if min_version is not None and Version(module.__version__) < Version(min_version):
        raise ValueError(
            f"Requires {dist_name} version {min_version} or higher, but you have {module.__version__}. "
            f"Upgrade with: pip install -U {dist_name}"
        )
    return module


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
    batch_size: int = 100,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Embed strings with a Gemini embedding model through the `google-genai` SDK.

    :param strings: List of strings.
    :param model: Gemini embedding model, e.g. "gemini-embedding-001".
    :param api_key: Gemini API key.
    :param return_numpy: If True, return a numpy array; else a torch tensor.
    :param batch_size: Number of strings per API request (the API accepts at most 100).
    :return: Embeddings, one row per string.
    """
    genai = _require("google.genai")
    client = genai.Client(api_key=api_key)
    config = genai.types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    model_name = _normalize_gemini_model_name(model)

    vectors: List[List[float]] = []
    for start in range(0, len(strings), batch_size):
        batch = strings[start:start + batch_size]
        response = client.models.embed_content(model=model_name, contents=batch, config=config)
        if response.embeddings is None or len(response.embeddings) != len(batch):
            raise ValueError("Gemini embedding response does not contain one embedding per input string")
        vectors.extend(embedding.values for embedding in response.embeddings)

    if return_numpy:
        return np.array(vectors, dtype=np.float32)
    return torch.tensor(vectors, dtype=torch.float32)

def load_model(model_path: str) -> LinkTransformer:
    """
    Load a saved LinkTransformer model.

    :param model_path: Path to the saved model.
    :return: The loaded LinkTransformer model.
    """
    try:
        return LinkTransformer(model_path)
    except Exception as exc:
        raise ValueError(
            f"Can't load the model {model_path!r}; please check the path or Hugging Face model id. "
            "All transformer based models from Hugging Face (https://huggingface.co/) are supported. "
            "We recommend sentence-transformers models (https://www.sbert.net/docs/pretrained_models.html) "
            "as they are trained for semantic similarity tasks."
        ) from exc


def load_clf(model_path: str, num_labels: int = 2) -> transformers.AutoModelForSequenceClassification:
    """
    Load the classification model.

    :param model_path: Path to the saved model.
    :param num_labels: Number of classes to predict (default is two)
    :return: The loaded Classification model
    """
    try:
        return transformers.AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    except Exception as exc:
        raise ValueError(f"Error loading classification model {model_path!r}.") from exc


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

def _is_openai_embedding_model(model: Any) -> bool:
    if not isinstance(model, str):
        return False
    model_name = model.lower()
    return "text-embedding" in model_name or "ada-002" in model_name


# Separator for multi-column text sent to API embedding models, which have no Hugging Face tokenizer
API_SEP_TOKEN = "<SEP>"


def serialize_columns(df: pd.DataFrame, columns: list, sep_token: Optional[str] = None, model: str = None) -> list:
    """
    Serialize columns of a DataFrame into a single string.

    :param df: The DataFrame.
    :param columns: The columns to serialize.
    :param sep_token: The token to use to separate columns. If None, it is chosen from the model: the
        tokenizer's sep token for Hugging Face models, "<SEP>" for OpenAI/Gemini embedding models, and
        "</s>" when no tokenizer is found.
    :param model: The language model whose separator token to use (optional).
    :return: List of serialized strings.
    """

    def _load_sep_token(model_name: str) -> Optional[str]:
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            return tokenizer.sep_token
        except Exception:
            return None

    if sep_token is None:
        if isinstance(model, LinkTransformer):
            sep_token = model.tokenizer.sep_token
        elif isinstance(model, str) and (_is_openai_embedding_model(model) or _is_gemini_embedding_model(model)):
            sep_token = API_SEP_TOKEN
        elif isinstance(model, str):
            candidates = [model] if "/" in model else [model, f"sentence-transformers/{model}", f"linktransformers/{model}"]
            for candidate in candidates:
                sep_token = _load_sep_token(candidate)
                if sep_token is not None:
                    break
            else:
                logger.warning("Could not load tokenizer for serialization. Using default sep token of </s>")

    if not isinstance(sep_token, str) or sep_token == "":
        sep_token = "</s>"

    return df[columns].apply(
        lambda x: sep_token.join(["" if pd.isna(v) else str(v) for v in x.tolist()]),
        axis=1,
    ).tolist()


_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "vocab.txt", "spiece.model",
                    "sentencepiece.bpe.model", "tokenizer.model")


def _base_model_name(model_path: str) -> Optional[str]:
    """The base model a local checkpoint was fine-tuned from, if the checkpoint records it."""
    lt_config_path = os.path.join(model_path, "LT_training_config.json")
    if os.path.isfile(lt_config_path):  # written by train_clf_model
        with open(lt_config_path, encoding="utf-8") as f:
            base = json.load(f).get("base_model_path")
        if base:
            return base
    config_path = os.path.join(model_path, "config.json")
    if os.path.isfile(config_path):  # older transformers versions save _name_or_path
        with open(config_path, encoding="utf-8") as f:
            base = json.load(f).get("_name_or_path")
        if base and os.path.normpath(base) != os.path.normpath(model_path):
            return base
    return None


def _load_tokenizer(model_path: str, tokenizer: Optional[str] = None):
    """
    Load the tokenizer for a (classification) model.

    Checkpoints saved by a plain transformers Trainer often have no tokenizer files. Recent transformers
    versions then build an empty tokenizer without raising, so local checkpoints are checked for tokenizer
    files and fall back to the base model's tokenizer.

    :param model_path: Model path or Hugging Face model id.
    :param tokenizer: Optional tokenizer path or id to use instead.
    """
    if tokenizer is not None:
        return transformers.AutoTokenizer.from_pretrained(tokenizer)
    if os.path.isdir(model_path) and not any(os.path.exists(os.path.join(model_path, f)) for f in _TOKENIZER_FILES):
        base_model = _base_model_name(model_path)
        if base_model is None:
            raise OSError(
                f"No tokenizer files found in {model_path!r} and the checkpoint does not record its base model. "
                "Pass the tokenizer of the base model, e.g. tokenizer='distilroberta-base'."
            )
        warnings.warn(f"No tokenizer files found in {model_path!r}; using the tokenizer of its base model {base_model!r}.")
        return transformers.AutoTokenizer.from_pretrained(base_model)
    return transformers.AutoTokenizer.from_pretrained(model_path)


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
                warnings.warn("Use of non-sentence transformer models is not recommended and is not actively supported. This will probably not work")
            embeddings = model.encode(strings, batch_size=batch_size,convert_to_numpy=return_numpy,convert_to_tensor=not return_numpy)
        else:
            raise ValueError(f"Invalid model type: {type(model)}")
    else:
        openai = _require("openai", min_version="1.0.0")
        client = openai.OpenAI(api_key=openai_key)
        # OpenAI limits the tokens per request; group strings into batches of about 5000 characters
        batches, batch, batch_length = [], [], 0
        for string in strings:
            if batch and batch_length + len(string) > 5000:
                batches.append(batch)
                batch, batch_length = [], 0
            batch.append(string)
            batch_length += len(string)
        if batch:
            batches.append(batch)

        embeddings = []
        for batch in batches:
            values = [item.embedding for item in client.embeddings.create(input=batch, model=model).data]
            if return_numpy:
                embeddings.append(np.asarray(values, dtype=np.float32))
            else:  # prep as tensor
                embeddings.append(torch.tensor(values, dtype=torch.float32))
        if return_numpy:
            embeddings = np.concatenate(embeddings, axis=0)
        else:
            embeddings = torch.cat(embeddings, dim=0)

    return embeddings



def tokenize_data_for_inference(corpus: str, name: str, hf_model: str):
    datasets = _require("datasets")
    dataset = datasets.Dataset.from_dict({name: corpus})

    # Instantiate tokenizer
    tokenizer = _load_tokenizer(hf_model)

    # Tokenize datasets
    def tokenize_function(dataset):
        return tokenizer(dataset[name], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    return tokenized_dataset

_CHAT_PARAMS = ("temperature", "max_tokens", "max_completion_tokens", "top_p", "frequency_penalty", "presence_penalty")


def _chat_completion(client, model: str, messages: list, params: Optional[Dict] = None, defaults: Optional[Dict] = None):
    """
    Create an OpenAI chat completion, adapting the sampling parameters to the model.

    `max_tokens` is sent as `max_completion_tokens` (newer models reject the former), and any parameter
    the API reports as unsupported (e.g. temperature or top_p for reasoning models) is dropped and the
    request retried.
    """
    kwargs = dict(defaults or {})
    kwargs.update({k: v for k, v in (params or {}).items() if k in _CHAT_PARAMS})
    if "max_tokens" in kwargs:
        kwargs.setdefault("max_completion_tokens", kwargs.pop("max_tokens"))
    openai = _require("openai")
    while True:
        try:
            return client.chat.completions.create(model=model, messages=messages, **kwargs)
        except openai.BadRequestError as exc:
            rejected = getattr(exc, "param", None)
            if rejected not in kwargs:
                raise
            logger.info(f"Model {model} does not support `{rejected}`; retrying without it.")
            kwargs.pop(rejected)


def _is_retryable(exc: Exception) -> bool:
    """Rate limits, timeouts, connection and server errors are worth retrying; other client errors are not."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return not (isinstance(status, int) and 400 <= status < 500 and status not in (408, 409, 429))


def _match_label(text: str, label_dict: dict):
    """Map a model answer such as "Yes." or " yes" to a key of label_dict; None if it matches no label."""
    normalized = {str(k).strip().lower(): v for k, v in label_dict.items()}
    answer = (text or "").strip().strip(".!\"'").strip().lower()
    return normalized.get(answer)


def get_completion_from_messages(
        client: "openai.OpenAI",
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

    # No default token limit: reasoning models spend completion tokens before answering
    response = _chat_completion(
        client,
        model,
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}],
        params=openai_params,
        defaults={"temperature": 0},
    )
    return response.choices[0].message.content, response.usage.total_tokens
    

def predict_rows_with_openai(
        strings_col: List[str],
        model: str = "gpt-4o-mini",
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

    openai = _require("openai", min_version="1.0.0")
    openai_params = openai_params or {}

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
            except Exception as e:  # retry rate limits, timeouts and server errors only
                if not _is_retryable(e) or num_retry == max_retries - 1:
                    raise
                tqdm.write(
                    f"Encountered error: {repr(e)}. Will retry after sleeping for {ratelimit_sleep_time * (2 ** num_retry)} seconds (attempt {num_retry + 1}/{max_retries})")
                time.sleep(ratelimit_sleep_time * (2 ** num_retry))

    if label_dict is None:
        label_dict = {"Yes": 1, "No": 0}

    preds_numeric = [_match_label(x, label_dict) for x in preds]
    if any(p is None for p in preds_numeric):
        warnings.warn(
            "Failed to convert OpenAI text labels to numeric labels. Text labels are kept. \
            You may want to modify the prompt or the label dict. ")
        return preds
    return preds_numeric
    

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
            resp = _chat_completion(client, model, messages=[system_msg, user_msg], params=extra)
            out = resp.choices[0].message.content
            arr = json.loads(out)
            if not isinstance(arr, list):
                raise ValueError("expected JSON list")
            if len(arr) != len(texts):
                raise ValueError(f"expected {len(texts)} items, got {len(arr)}: {arr!r} for inputs {texts!r}")
            
            return arr
        except Exception as e:
            if attempt == retries - 1 or not _is_retryable(e):
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
