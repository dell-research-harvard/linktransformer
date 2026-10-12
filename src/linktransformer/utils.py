import os
import re
import time
import warnings
import threading
import concurrent.futures
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
from linktransformer.api_runtime import (
    append_jsonl,
    call_with_retry,
    log_failure,
    parse_retry_policy,
    read_jsonl,
)


def _is_gemini_embedding_model(model: Any) -> bool:
    if not isinstance(model, str):
        return False
    model_name = model.lower()
    return (
        "gemini" in model_name
        or "text-embedding-004" in model_name
        or "embedding-001" in model_name
    )


def _is_openai_embedding_model(model: Any) -> bool:
    if not isinstance(model, str):
        return False
    model_name = model.lower()
    return (
        "text-embedding" in model_name
        or "ada-002" in model_name
    )


def _is_gemini_chat_model(model: Any) -> bool:
    return isinstance(model, str) and "gemini" in model.lower()


def _resolve_llm_provider(
    model: Optional[str],
    llm_provider: str,
    openai_key: Optional[str],
    gemini_key: Optional[str],
) -> str:
    if llm_provider not in {"auto", "openai", "gemini"}:
        raise ValueError("llm_provider must be one of {'auto', 'openai', 'gemini'}")

    if llm_provider != "auto":
        return llm_provider

    if _is_gemini_chat_model(model) or gemini_key or os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if openai_key or os.getenv("OPENAI_API_KEY"):
        return "openai"
    raise ValueError(
        "Could not resolve llm provider automatically. Provide openai_key/OPENAI_API_KEY or gemini_key/GEMINI_API_KEY, "
        "or pass llm_provider explicitly."
    )


def _extract_first_json_array(text: str) -> List[Any]:
    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            return payload
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        payload = json.loads(fenced.group(1))
        if isinstance(payload, list):
            return payload

    bracketed = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if bracketed:
        payload = json.loads(bracketed.group(0))
        if isinstance(payload, list):
            return payload

    raise ValueError("expected JSON list")


def _canonicalize_label(raw_label: Any, label_dict: Dict[Any, Any]) -> Any:
    if raw_label in label_dict:
        return raw_label

    if isinstance(raw_label, str):
        raw_norm = raw_label.strip().lower()
        for key in label_dict.keys():
            if isinstance(key, str) and key.strip().lower() == raw_norm:
                return key

    return raw_label


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
        return LinkTransformer(model_path)
    except Exception as exc:
        raise ValueError(
            "Can't load the model, please check your path. All transformer-based models from HuggingFace are supported. "
            "We recommend sentence-transformers models for semantic similarity tasks."
        ) from exc


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

    user_provided_sep_token = isinstance(sep_token, str) and sep_token not in {"", "</s>"}

    ###if model is string
    if isinstance(model, str):
        # For API embedding models (OpenAI/Gemini), do not probe HuggingFace tokenizer.
        if _is_openai_embedding_model(model) or _is_gemini_embedding_model(model):
            if not isinstance(sep_token, str) or sep_token == "":
                sep_token = "</s>"
        # If user explicitly passed a separator, respect it and skip tokenizer lookup.
        elif user_provided_sep_token:
            pass
        elif model is not None:
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
                     sbert: bool = True, openai_key: str = None, gemini_key: str = None, return_numpy=True,
                     checkpoint_path: Optional[str] = None, resume: bool = False,
                     failure_log_path: Optional[str] = None,
                     retry_policy: Optional[Dict[str, Any]] = None) -> Union[np.ndarray, torch.Tensor]:
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
        policy = parse_retry_policy(retry_policy, default_max_retries=5, default_base_sleep=15.0)
        if checkpoint_path is not None:
            cached = {}
            if resume:
                for rec in read_jsonl(checkpoint_path):
                    if rec.get("kind") == "gemini_embedding_item" and "index" in rec and "embedding" in rec:
                        cached[int(rec["index"])] = rec["embedding"]

            vectors: List[np.ndarray] = []
            for index, text in enumerate(strings):
                if index in cached:
                    vectors.append(np.array(cached[index], dtype=np.float32))
                    continue

                def _gemini_call() -> np.ndarray:
                    item = infer_embeddings_with_gemini([text], model=model, api_key=api_key, return_numpy=True)
                    return item[0]

                try:
                    vector = call_with_retry(_gemini_call, policy)
                except Exception as exc:
                    log_failure(
                        failure_log_path,
                        function_name="infer_embeddings",
                        provider="gemini",
                        model=str(model),
                        item_key=index,
                        attempt=policy.max_retries,
                        error=exc,
                    )
                    raise

                append_jsonl(
                    checkpoint_path,
                    {
                        "kind": "gemini_embedding_item",
                        "index": index,
                        "embedding": vector.tolist(),
                    },
                )
                vectors.append(np.array(vector, dtype=np.float32))

            matrix = np.vstack(vectors).astype(np.float32)
            if return_numpy:
                return matrix
            return torch.tensor(matrix, dtype=torch.float32)

        try:
            result = call_with_retry(
                lambda: infer_embeddings_with_gemini(strings, model=model, api_key=api_key, return_numpy=return_numpy),
                policy,
            )
            return result
        except Exception as exc:
            log_failure(
                failure_log_path,
                function_name="infer_embeddings",
                provider="gemini",
                model=str(model),
                item_key="all",
                attempt=policy.max_retries,
                error=exc,
            )
            raise

    if _is_openai_embedding_model(model):
        resolved_openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        if resolved_openai_key is None:
            raise ValueError(
                "OpenAI embedding models require an API key. Pass your key via `openai_key` or set `OPENAI_API_KEY`."
            )
        openai_key = resolved_openai_key

    if openai_key is None:
        encode_kwargs = {
            "batch_size": batch_size,
            "convert_to_numpy": return_numpy,
            "convert_to_tensor": not return_numpy,
            "show_progress_bar": True,
        }

        if isinstance(model, LinkTransformer):
            try:
                embeddings = model.encode(strings, **encode_kwargs)
            except TypeError:
                encode_kwargs.pop("show_progress_bar", None)
                embeddings = model.encode(strings, **encode_kwargs)
        elif isinstance(model, str):
            if sbert:
                model = LinkTransformer(model)
            else:
                model = transformers.AutoModel.from_pretrained(model)
                print("Warning: Use of non-sentence transformer models is not recommended and is not actively supported. This will probably not work")
            try:
                embeddings = model.encode(strings, **encode_kwargs)
            except TypeError:
                encode_kwargs.pop("show_progress_bar", None)
                embeddings = model.encode(strings, **encode_kwargs)
        else:
            raise ValueError(f"Invalid model type: {type(model)}")
    else:
        openai.api_key = openai_key
        policy = parse_retry_policy(retry_policy, default_max_retries=5, default_base_sleep=15.0)
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
        split_ranges = [(split_indices[i], split_indices[i + 1]) for i in range(len(split_indices) - 1)]

        chunk_cache = {}
        if checkpoint_path is not None and resume:
            for rec in read_jsonl(checkpoint_path):
                if rec.get("kind") == "openai_embedding_chunk" and "chunk_start" in rec and "chunk_end" in rec and "embeddings" in rec:
                    chunk_cache[(int(rec["chunk_start"]), int(rec["chunk_end"]))] = rec["embeddings"]

        ##Get the embeddings for each of the split lists
        embeddings = []
        for i in range(len(split_strings)):
            chunk_start, chunk_end = split_ranges[i]
            chunk_key = (chunk_start, chunk_end)

            if chunk_key in chunk_cache:
                cached_array = np.array(chunk_cache[chunk_key], dtype=np.float32)
                if return_numpy:
                    embeddings.append(cached_array)
                else:
                    embeddings.append(torch.tensor(cached_array, dtype=torch.float32))
                continue

            def _openai_embeddings_call():
                if not openai.__version__ >= "1.0.0":
                    response = openai.embeddings.create(input=split_strings[i], model=model)["data"]
                    f = lambda x: x["embedding"]
                else:
                    response = openai.embeddings.create(input=split_strings[i], model=model).data
                    f = lambda x: x.embedding
                return np.array(list(map(f, response)), dtype=np.float32)

            try:
                chunk_vectors = call_with_retry(_openai_embeddings_call, policy)
            except Exception as exc:
                log_failure(
                    failure_log_path,
                    function_name="infer_embeddings",
                    provider="openai",
                    model=str(model),
                    item_key=f"{chunk_start}:{chunk_end}",
                    attempt=policy.max_retries,
                    error=exc,
                )
                raise

            append_jsonl(
                checkpoint_path,
                {
                    "kind": "openai_embedding_chunk",
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_end,
                    "embeddings": chunk_vectors.tolist(),
                },
            )
            if return_numpy:
                embeddings.append(chunk_vectors)
            else:
                embeddings.append(torch.tensor(chunk_vectors, dtype=torch.float32))
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


def _extract_unsupported_param_from_error(error_text: str) -> Optional[str]:
    patterns = [
        r"unsupported parameter(?:\s*[:=])?\s*['\"]?([a-zA-Z_][a-zA-Z0-9_]*)['\"]?",
        r"parameter\s+['\"]?([a-zA-Z_][a-zA-Z0-9_]*)['\"]?.*not supported",
        r"['\"]param['\"]\s*:\s*['\"]([a-zA-Z_][a-zA-Z0-9_]*)['\"]",
        r"unsupported value\s*:\s*['\"]?([a-zA-Z_][a-zA-Z0-9_]*)['\"]?",
    ]
    for pattern in patterns:
        match = re.search(pattern, error_text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _openai_chat_completion_resilient(client: openai.OpenAI, request_kwargs: Dict[str, Any]) -> Any:
    effective_kwargs: Dict[str, Any] = dict(request_kwargs)
    adjusted_params: List[str] = []
    completion_budget_bumped = False

    def _bump_completion_budget(min_tokens: int = 32) -> bool:
        nonlocal completion_budget_bumped
        if completion_budget_bumped:
            return False

        if "max_completion_tokens" in effective_kwargs:
            current = effective_kwargs.get("max_completion_tokens")
            if isinstance(current, int):
                effective_kwargs["max_completion_tokens"] = max(current, min_tokens)
            else:
                effective_kwargs["max_completion_tokens"] = min_tokens
            completion_budget_bumped = True
            adjusted_params.append("max_completion_tokens:bumped")
            return True

        if "max_tokens" in effective_kwargs:
            current = effective_kwargs.pop("max_tokens")
            if isinstance(current, int):
                effective_kwargs["max_completion_tokens"] = max(current, min_tokens)
            else:
                effective_kwargs["max_completion_tokens"] = min_tokens
            completion_budget_bumped = True
            adjusted_params.append("max_tokens->max_completion_tokens:bumped")
            return True

        return False

    while True:
        try:
            response = client.chat.completions.create(**effective_kwargs)
            break
        except Exception as exc:
            error_text = str(exc)
            unsupported_param = _extract_unsupported_param_from_error(error_text)

            if unsupported_param == "max_tokens" and "max_tokens" in effective_kwargs:
                max_tokens_val = effective_kwargs.pop("max_tokens")
                effective_kwargs["max_completion_tokens"] = max_tokens_val
                adjusted_params.append("max_tokens->max_completion_tokens")
                continue

            if unsupported_param and unsupported_param in effective_kwargs:
                effective_kwargs.pop(unsupported_param, None)
                adjusted_params.append(unsupported_param)
                continue

            if "use 'max_completion_tokens' instead" in error_text.lower() and "max_tokens" in effective_kwargs:
                max_tokens_val = effective_kwargs.pop("max_tokens")
                effective_kwargs["max_completion_tokens"] = max_tokens_val
                adjusted_params.append("max_tokens->max_completion_tokens")
                continue

            if (
                "max_tokens or model output limit was reached" in error_text.lower()
                or "could not finish the message because max_tokens" in error_text.lower()
            ):
                if _bump_completion_budget():
                    continue

            raise

    if adjusted_params:
        warnings.warn(
            f"OpenAI call adjusted unsupported params: {sorted(set(adjusted_params))}",
            UserWarning,
        )

    return response


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

    request_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": sys_prompt
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "temperature": openai_params["temperature"] if "temperature" in openai_params else 0,
        "max_tokens": openai_params["max_tokens"] if "max_tokens" in openai_params else 100,
        "top_p": openai_params["top_p"] if "top_p" in openai_params else 0,
        "frequency_penalty": openai_params["frequency_penalty"] if "frequency_penalty" in openai_params else 0,
        "presence_penalty": openai_params["presence_penalty"] if "presence_penalty" in openai_params else 0,
    }
    response = _openai_chat_completion_resilient(client, request_kwargs)
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
        ratelimit_sleep_time: int = 15,
        checkpoint_path: Optional[str] = None,
        resume: bool = False,
        failure_log_path: Optional[str] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
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

    openai_params = {} if openai_params is None else openai_params
    policy = parse_retry_policy(
        retry_policy,
        default_max_retries=max_retries,
        default_base_sleep=float(ratelimit_sleep_time),
    )

    raw_preds: Dict[int, str] = {}
    if checkpoint_path is not None and resume:
        for rec in read_jsonl(checkpoint_path):
            if rec.get("kind") == "openai_classification_row" and "index" in rec and "response" in rec:
                raw_preds[int(rec["index"])] = rec["response"]

    # create client
    client = openai.OpenAI(
        api_key=openai_key,
        timeout=openai_params["request_timeout"] if "request_timeout" in openai_params else 10
    )

    # get results from api
    for i in tqdm(range(len(strings_col))):
        if i in raw_preds:
            continue

        def _row_call() -> str:
            response_text, _ = get_completion_from_messages(
                client, strings_col[i], model, openai_key, openai_topic, openai_prompt, openai_params
            )
            return response_text

        try:
            row_prediction = call_with_retry(_row_call, policy)
        except Exception as exc:
            log_failure(
                failure_log_path,
                function_name="predict_rows_with_openai",
                provider="openai",
                model=model,
                item_key=i,
                attempt=policy.max_retries,
                error=exc,
            )
            raise

        raw_preds[i] = row_prediction
        append_jsonl(
            checkpoint_path,
            {
                "kind": "openai_classification_row",
                "index": i,
                "response": row_prediction,
            },
        )

    preds = [raw_preds[i] for i in range(len(strings_col))]

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


def predict_rows_with_llm(
        strings_col: List[str],
        model: str,
        openai_key: Optional[str] = None,
        gemini_key: Optional[str] = None,
        llm_provider: str = "auto",
        openai_topic: Optional[str] = None,
        openai_prompt: Optional[str] = None,
        openai_params: Optional[dict] = None,
        label_dict: Optional[dict] = None,
        max_retries: int = 5,
        ratelimit_sleep_time: int = 15,
        checkpoint_path: Optional[str] = None,
        resume: bool = False,
        failure_log_path: Optional[str] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        batch_size: int = 1,
        throttle_seconds: float = 0.0,
        checkpoint_window_size: Optional[int] = None,
        workers: int = 1,
        progress_bar: bool = True,
):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if checkpoint_window_size is not None and checkpoint_window_size < 1:
        raise ValueError("checkpoint_window_size must be >= 1")
    if workers < 1:
        raise ValueError("workers must be >= 1")

    checkpoint_window = checkpoint_window_size or batch_size

    openai_params = {} if openai_params is None else openai_params
    label_dict = {"Yes": 1, "No": 0} if label_dict is None else label_dict
    provider = _resolve_llm_provider(model, llm_provider, openai_key, gemini_key)

    if provider == "openai" and openai.__version__ < "1.0.0":
        raise ValueError(f"Requires OpenAI API version 1.0.0 or higher, but you have {openai.__version__}")

    policy = parse_retry_policy(
        retry_policy,
        default_max_retries=max_retries,
        default_base_sleep=float(ratelimit_sleep_time),
    )

    resolved_openai_key = openai_key or os.getenv("OPENAI_API_KEY")
    resolved_gemini_key = gemini_key or os.getenv("GEMINI_API_KEY")

    if provider == "openai" and resolved_openai_key is None:
        raise ValueError("OpenAI classification requires `openai_key` or OPENAI_API_KEY.")
    if provider == "gemini" and resolved_gemini_key is None:
        raise ValueError("Gemini classification requires `gemini_key` or GEMINI_API_KEY.")

    if openai_prompt is None:
        if openai_topic is None:
            raise ValueError("Must provide either openai_topic or openai_prompt for LLM classification")
        base_prompt = f"Determine whether each text is about {openai_topic}."
    else:
        base_prompt = openai_prompt

    allowed_labels = list(label_dict.keys())
    label_instr = (
        "\n\nClassify each input independently. Return only a JSON array of labels, one per input, "
        f"in order, using only these labels exactly: {json.dumps(allowed_labels)}."
    )
    effective_prompt = base_prompt + label_instr

    raw_preds: Dict[int, Any] = {}
    if checkpoint_path is not None and resume:
        for rec in read_jsonl(checkpoint_path):
            if rec.get("kind") == "llm_classification_row" and "index" in rec and "response" in rec:
                raw_preds[int(rec["index"])] = rec["response"]
            elif rec.get("kind") == "openai_classification_row" and "index" in rec and "response" in rec:
                raw_preds[int(rec["index"])] = rec["response"]
            elif rec.get("kind") == "llm_classification_batch" and "start" in rec and "responses" in rec:
                responses = rec.get("responses", [])
                indices = rec.get("indices")
                if isinstance(indices, list) and isinstance(responses, list):
                    for idx_raw, value in zip(indices, responses):
                        idx = int(idx_raw)
                        if 0 <= idx < len(strings_col) and idx not in raw_preds:
                            raw_preds[idx] = value
                elif isinstance(responses, list):
                    batch_start = int(rec["start"])
                    for offset, value in enumerate(responses):
                        idx = batch_start + offset
                        if 0 <= idx < len(strings_col) and idx not in raw_preds:
                            raw_preds[idx] = value

    checkpoint_buffer_indices: List[int] = []
    checkpoint_buffer_values: List[Any] = []

    def _flush_checkpoint_buffer(*, force: bool = False) -> None:
        if checkpoint_path is None:
            return

        while len(checkpoint_buffer_indices) >= checkpoint_window or (force and len(checkpoint_buffer_indices) > 0):
            flush_n = checkpoint_window if len(checkpoint_buffer_indices) >= checkpoint_window else len(checkpoint_buffer_indices)
            flush_indices = checkpoint_buffer_indices[:flush_n]
            flush_values = checkpoint_buffer_values[:flush_n]

            append_jsonl(
                checkpoint_path,
                {
                    "kind": "llm_classification_batch",
                    "start": min(flush_indices),
                    "end": max(flush_indices) + 1,
                    "indices": flush_indices,
                    "responses": flush_values,
                    "provider": provider,
                },
            )

            del checkpoint_buffer_indices[:flush_n]
            del checkpoint_buffer_values[:flush_n]

    thread_local = threading.local()

    def _get_openai_client():
        client = getattr(thread_local, "openai_client", None)
        if client is None:
            client = openai.OpenAI(
                api_key=resolved_openai_key,
                timeout=openai_params.get("request_timeout", 10),
            )
            thread_local.openai_client = client
        return client

    def _get_gemini_model_client():
        model_client = getattr(thread_local, "gemini_model_client", None)
        if model_client is not None:
            return model_client

        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "Gemini classification requires `google-generativeai`. Install it with: pip install google-generativeai"
            ) from exc

        genai.configure(api_key=resolved_gemini_key)
        model_client = genai.GenerativeModel(model_name=model)
        thread_local.gemini_model_client = model_client
        return model_client

    pending_batches: List[Tuple[int, int, List[int], List[str]]] = []
    for start in range(0, len(strings_col), batch_size):
        end = min(start + batch_size, len(strings_col))
        indices = list(range(start, end))
        if all(i in raw_preds for i in indices):
            continue
        pending_indices = [i for i in indices if i not in raw_preds]
        batch_texts = [strings_col[i] for i in pending_indices]
        if len(pending_indices) > 0:
            pending_batches.append((start, end, pending_indices, batch_texts))

    progress = tqdm(total=len(pending_batches), disable=not progress_bar, desc="classifying rows")

    def _run_batch(batch_item: Tuple[int, int, List[int], List[str]]) -> Tuple[int, int, List[int], List[Any]]:
        start, end, pending_indices, batch_texts = batch_item

        if throttle_seconds > 0:
            time.sleep(throttle_seconds)

        def _batch_call() -> List[Any]:
            if provider == "openai":
                request_kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": effective_prompt},
                        {"role": "user", "content": json.dumps(batch_texts)},
                    ],
                    **{k: openai_params[k] for k in ("temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty") if k in openai_params},
                }
                response = _openai_chat_completion_resilient(_get_openai_client(), request_kwargs)
                content = response.choices[0].message.content or "[]"
            else:
                generation_config: Dict[str, Any] = {}
                if "temperature" in openai_params:
                    generation_config["temperature"] = openai_params["temperature"]
                if "top_p" in openai_params:
                    generation_config["top_p"] = openai_params["top_p"]
                if "top_k" in openai_params:
                    generation_config["top_k"] = openai_params["top_k"]
                if "max_output_tokens" in openai_params:
                    generation_config["max_output_tokens"] = openai_params["max_output_tokens"]
                elif "max_tokens" in openai_params:
                    generation_config["max_output_tokens"] = openai_params["max_tokens"]

                prompt_text = f"{effective_prompt}\n\nInputs:\n{json.dumps(batch_texts)}"
                gemini_model_client = _get_gemini_model_client()
                if generation_config:
                    response = gemini_model_client.generate_content(prompt_text, generation_config=generation_config)
                else:
                    response = gemini_model_client.generate_content(prompt_text)
                content = getattr(response, "text", None) or "[]"

            arr = _extract_first_json_array(content)
            if len(arr) != len(batch_texts):
                raise ValueError(f"expected {len(batch_texts)} labels, got {len(arr)}")
            return arr

        try:
            batch_labels = call_with_retry(_batch_call, policy)
        except Exception as exc:
            log_failure(
                failure_log_path,
                function_name="predict_rows_with_llm",
                provider=provider,
                model=model,
                item_key=f"{start}:{end}",
                attempt=policy.max_retries,
                error=exc,
            )
            raise

        return start, end, pending_indices, batch_labels

    try:
        if workers == 1:
            for batch_item in pending_batches:
                _, _, pending_indices, batch_labels = _run_batch(batch_item)
                for idx, value in zip(pending_indices, batch_labels):
                    raw_preds[idx] = value
                    checkpoint_buffer_indices.append(idx)
                    checkpoint_buffer_values.append(value)
                _flush_checkpoint_buffer(force=False)
                progress.update(1)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_run_batch, batch_item) for batch_item in pending_batches]
                for fut in concurrent.futures.as_completed(futures):
                    _, _, pending_indices, batch_labels = fut.result()
                    for idx, value in zip(pending_indices, batch_labels):
                        raw_preds[idx] = value
                        checkpoint_buffer_indices.append(idx)
                        checkpoint_buffer_values.append(value)
                    _flush_checkpoint_buffer(force=False)
                    progress.update(1)
    finally:
        progress.close()

    _flush_checkpoint_buffer(force=True)

    preds = [raw_preds[i] for i in range(len(strings_col))]

    try:
        canonical_preds = [_canonicalize_label(x, label_dict) for x in preds]
        preds_numeric = [label_dict[x] for x in canonical_preds]
        return preds_numeric
    except Exception:
        warnings.warn(
            "Failed to convert LLM text labels to numeric labels. Text labels are kept. "
            "You may want to modify the prompt or the label dict."
        )
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
    failure_log_path = params.get("failure_log_path")
    custom_retry_policy = params.get("retry_policy")

    # append JSON instructions automatically
    json_instr = (
        "\n\nPlease return only a JSON array of strings, "
        "one entry per input, in the same order, with no extra text."
    )
    effective_prompt = prompt + json_instr

    system_msg = {"role": "system", "content": effective_prompt}
    user_msg   = {"role": "user",   "content": json.dumps(texts)}


    policy = parse_retry_policy(
        custom_retry_policy,
        default_max_retries=retries,
        default_base_sleep=float(backoff),
    )

    def _call_transform() -> List[str]:
            request_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": [system_msg, user_msg],
                **{k: extra[k] for k in ("temperature", "max_tokens", "top_p",
                                         "frequency_penalty", "presence_penalty")
                   if k in extra}
            }
            resp = _openai_chat_completion_resilient(client, request_kwargs)
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

    try:
        return call_with_retry(_call_transform, policy)
    except Exception as exc:
        log_failure(
            failure_log_path,
            function_name="openai_transform",
            provider="openai",
            model=model,
            item_key=f"batch_size={len(texts)}",
            attempt=policy.max_retries,
            error=exc,
        )
        raise


def gemini_transform(
    texts: List[str],
    params: Dict[str, Any]
) -> List[str]:
    model = params["model"]
    prompt = params["prompt"]
    retries = params.get("max_retries", 5)
    backoff = params.get("ratelimit_sleep_time", 15)
    extra = params.get("openai_params", {})
    failure_log_path = params.get("failure_log_path")
    custom_retry_policy = params.get("retry_policy")
    model_client = params["gemini_model"]

    json_instr = (
        "\n\nPlease return only a JSON array of strings, "
        "one entry per input, in the same order, with no extra text."
    )
    effective_prompt = prompt + json_instr

    generation_config: Dict[str, Any] = {}
    if "temperature" in extra:
        generation_config["temperature"] = extra["temperature"]
    if "top_p" in extra:
        generation_config["top_p"] = extra["top_p"]
    if "top_k" in extra:
        generation_config["top_k"] = extra["top_k"]
    if "max_output_tokens" in extra:
        generation_config["max_output_tokens"] = extra["max_output_tokens"]
    elif "max_tokens" in extra:
        generation_config["max_output_tokens"] = extra["max_tokens"]

    policy = parse_retry_policy(
        custom_retry_policy,
        default_max_retries=retries,
        default_base_sleep=float(backoff),
    )

    def _call_transform() -> List[str]:
        prompt_text = f"{effective_prompt}\n\nInputs:\n{json.dumps(texts)}"
        if generation_config:
            resp = model_client.generate_content(prompt_text, generation_config=generation_config)
        else:
            resp = model_client.generate_content(prompt_text)

        out = getattr(resp, "text", None) or "[]"
        arr = _extract_first_json_array(out)
        if len(arr) != len(texts):
            raise ValueError(f"expected {len(texts)} items, got {len(arr)}")
        return [str(item) for item in arr]

    try:
        return call_with_retry(_call_transform, policy)
    except Exception as exc:
        log_failure(
            failure_log_path,
            function_name="gemini_transform",
            provider="gemini",
            model=model,
            item_key=f"batch_size={len(texts)}",
            attempt=policy.max_retries,
            error=exc,
        )
        raise
            
def transform_column(
    df: pd.DataFrame,
    column: str,
    transform_fn: Callable[[List[str], Dict[str,Any]], List[str]],
    fn_kwargs: Dict[str,Any],
    *,
    chunk_size: int = 50,
    output_column: Optional[str] = None,
    progress_bar: bool = True,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    failure_log_path: Optional[str] = None,
    retry_policy: Optional[Dict[str, Any]] = None,
    throttle_seconds: float = 0.0,
    checkpoint_window_size: Optional[int] = None,
    workers: int = 1,
) -> pd.DataFrame:
    """
    Generic: takes one column of strings, runs them through transform_fn in chunks,
    and appends a new column of the transforms.
    """
    if column not in df.columns:
        raise KeyError(column)
    if checkpoint_window_size is not None and checkpoint_window_size < 1:
        raise ValueError("checkpoint_window_size must be >= 1")
    if workers < 1:
        raise ValueError("workers must be >= 1")

    checkpoint_window = checkpoint_window_size or chunk_size
    ser = df[column].astype(str).tolist()
    transformed: List[Optional[str]] = [None] * len(ser)

    if checkpoint_path is not None and resume:
        for rec in read_jsonl(checkpoint_path):
            if rec.get("kind") == "transform_column_row" and "index" in rec and "value" in rec:
                idx = int(rec["index"])
                if 0 <= idx < len(transformed):
                    transformed[idx] = rec["value"]
            elif rec.get("kind") == "transform_column_batch" and "start" in rec and "values" in rec:
                values = rec.get("values", [])
                indices = rec.get("indices")
                if isinstance(indices, list) and isinstance(values, list):
                    for idx_raw, value in zip(indices, values):
                        idx = int(idx_raw)
                        if 0 <= idx < len(transformed) and transformed[idx] is None:
                            transformed[idx] = value
                elif isinstance(values, list):
                    start = int(rec["start"])
                    for offset, value in enumerate(values):
                        idx = start + offset
                        if 0 <= idx < len(transformed) and transformed[idx] is None:
                            transformed[idx] = value

    checkpoint_buffer_indices: List[int] = []
    checkpoint_buffer_values: List[Any] = []

    def _flush_checkpoint_buffer(*, force: bool = False) -> None:
        if checkpoint_path is None:
            return

        while len(checkpoint_buffer_indices) >= checkpoint_window or (force and len(checkpoint_buffer_indices) > 0):
            flush_n = checkpoint_window if len(checkpoint_buffer_indices) >= checkpoint_window else len(checkpoint_buffer_indices)
            flush_indices = checkpoint_buffer_indices[:flush_n]
            flush_values = checkpoint_buffer_values[:flush_n]

            append_jsonl(
                checkpoint_path,
                {
                    "kind": "transform_column_batch",
                    "start": min(flush_indices),
                    "end": max(flush_indices) + 1,
                    "indices": flush_indices,
                    "values": flush_values,
                },
            )

            del checkpoint_buffer_indices[:flush_n]
            del checkpoint_buffer_values[:flush_n]

    batch_ranges: List[Tuple[int, int]] = []
    for start in range(0, len(ser), chunk_size):
        end = min(start + chunk_size, len(ser))
        if all(transformed[i] is not None for i in range(start, end)):
            continue
        batch_ranges.append((start, end))

    progress = tqdm(total=len(batch_ranges), disable=not progress_bar, desc="applying transform")

    def _run_transform_batch(batch_range: Tuple[int, int]) -> Tuple[int, int, List[Any]]:
        start, end = batch_range
        batch = ser[start:end]
        call_kwargs = dict(fn_kwargs)
        call_kwargs["failure_log_path"] = failure_log_path
        call_kwargs["retry_policy"] = retry_policy
        if throttle_seconds > 0:
            time.sleep(throttle_seconds)

        try:
            batch_out = transform_fn(batch, call_kwargs)
        except Exception as exc:
            log_failure(
                failure_log_path,
                function_name="transform_column",
                provider="openai" if "client" in fn_kwargs else None,
                model=fn_kwargs.get("model"),
                item_key=f"{start}:{end}",
                attempt=1,
                error=exc,
            )
            raise

        if len(batch_out) != len(batch):
            raise ValueError("Transform function returned wrong batch size")

        return start, end, batch_out

    try:
        if workers == 1:
            for batch_range in batch_ranges:
                start, _, batch_out = _run_transform_batch(batch_range)
                for offset, value in enumerate(batch_out):
                    idx = start + offset
                    transformed[idx] = value
                    checkpoint_buffer_indices.append(idx)
                    checkpoint_buffer_values.append(value)
                _flush_checkpoint_buffer(force=False)
                progress.update(1)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_run_transform_batch, batch_range) for batch_range in batch_ranges]
                for fut in concurrent.futures.as_completed(futures):
                    start, _, batch_out = fut.result()
                    for offset, value in enumerate(batch_out):
                        idx = start + offset
                        transformed[idx] = value
                        checkpoint_buffer_indices.append(idx)
                        checkpoint_buffer_values.append(value)
                    _flush_checkpoint_buffer(force=False)
                    progress.update(1)
    finally:
        progress.close()

    _flush_checkpoint_buffer(force=True)

    if any(v is None for v in transformed):
        raise RuntimeError("Transform column did not produce outputs for all rows.")

    out_col = output_column or f"{column}_transformed"
    df = df.copy()
    df[out_col] = transformed
    return df
