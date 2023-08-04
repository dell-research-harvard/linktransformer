# LinkTransformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description

LinkTransformer is a Python library for merging and deduplicating dataframes using language model embeddings. It leverages popular Sentence Transformer (or any HuggingFace) models to generate embeddings for text data and provides functions to perform efficient 1:1, 1:m, and m:1 merges based on the similarity of embeddings. Additionally, the package includes utilities for clustering and data preprocessing. It also includes modifications to Sentence Transformers that allow for logging training runs on weights and biases.

## Features

- Merge dataframes using language model embeddings
- Deduplicate data based on similarity threshold
- Efficient 1:1, 1:m, and m:1 merges
- Clustering methods for grouping similar data
- Support for various NLP models available on HuggingFace

## Installation

```bash
pip install linktransformer
```

## Getting Started

```python

import linktransformer as lt
# Example usage of lm_merge_df
merged_df = lt.merge(df1, df2, merge_type='1:1', on='key_column', model='your-pretrained-model-from-huggingface')

# Example usage of dedup
deduplicated_df = lt.dedup_rows(df, model='your-pretrained-model', on='text_column', threshold=0.8)
```

All transformer based models from [HuggingFace](https://huggingface.co/) are supported. We recommend [sentence-transformers](https://www.sbert.net/docs/pretrained_models.html) for these tasks as they are trained for semantic similarity tasks. 

## Usage 

### Merging Pandas Dataframes

The merge function is used to merge two dataframes using language model embeddings. It supports three types of merges: 1:1, 1:m, and m:1. The function takes the following parameters:

```python
def merge(df1, df2, merge_type='1:1', on=None, model='your-pretrained-model', left_on=None, right_on=None, suffixes=('_x', '_y'),
                 use_gpu=False, batch_size=128, pooling_type='mean', openai_key=None):
    """
    Merge two dataframes using language model embeddings
    :param df1: first dataframe (left) 
    :param df2: second dataframe (right)
    :param merge_type: type of merge to perform 1:m or m:1 or 1:1
    :param model: language model to use
    :param on: column to join on in df1
    :param left_on: column to join on in df1
    :param right_on: column to join on in df2
    :param suffixes: suffixes to use for overlapping columns
    :return: merged dataframe
    """


```

A special case of merging is aggregation (use function: aggregate_rows)- when the left key is a list of items that need aggregation to the right keys. Semantic linking is also allowed with multiple columns as keys in both datasets. For larger datasets, merge_blocking can be used to merge within blocking keys. 


### Clustering or Deduplicating Data
```python
def dedup_rows(df, model, on, threshold=0.5, openai_key=None):
    """
    A function to deduplicate a dataframe based on a similarity threshold
    :param df: dataframe to deduplicate
    :param model: language model to use
    :param on: column to deduplicate on
    :param threshold: similarity threshold for clustering
    :param openai_key: Open ai key. 
    :return: deduplicated dataframe
    """

def cluster_rows(df, model, on, threshold=0.5, openai_key=None):
    """
    A function to deduplicate a dataframe based on a similarity threshold
    :param df: dataframe to deduplicate
    :param model: language model to use
    :param on: column to deduplicate on
    :param threshold: similarity threshold for clustering
    :param openai_key: Open ai key. 
    :return: deduplicated dataframe
    """

```

We allow a simple clustering function to cluster rows based on a key. Deduplication is just keeping only one row per cluster.


### Get similarity score between 2 sets of columns

```python

def evaluate_pairs(df,model,left_on,right_on,openai_key=None):
    """
    This function evaluates paired columns in a dataframe and gives a match score (cosine similarity). 
    Typically, this can be though of as a way to evaluate already merged in dataframes.

    :param df (DataFrame): Dataframe to evaluate.
    :param model (str): Language model to use.
    :param left_on (Union[str, List[str]]): Column(s) to evaluate on in df.
    :param right_on (Union[str, List[str]]): Reference column(s) to evaluate on in df.
    :return: DataFrame: The evaluated dataframe.
    """


```


### Training your own LinkTransformer model

```python


def train_model(
    model_path: str='your-pretrained-model',
    dataset_path: str = "data/es_mexican_products.xlsx",
    left_col_names: List[str] = ["description47"],
    right_col_names: List[str] = ['description48'],
    left_id_name: List[str] = ['tariffcode47'],
    right_id_name: List[str] = ['tariffcode48'],
    config_path: str = LINKAGE_CONFIG_PATH,
    training_args: dict = {"num_epochs":10},
    log_wandb: bool = False,
) -> str:
    """
    Train the LinkTransformer model.

    :param model_path (str): The name of the model to use.
    :param dataset_path (str): Path to the dataset in Excel format.
    :param left_col_names (List[str]): List of column names to use as left side data.
    :param right_col_names (List[str]): List of column names to use as right side data.
    :param left_id_name (List[str]): List of column names to use as identifiers for the left data.
    :param right_id_name (List[str]): List of column names to use as identifiers for the right data.
    :param config_path (str): Path to the JSON configuration file.
    :param training_args (dict): Dictionary of training arguments to override the config.
    :param log_wandb (bool): Whether to log the training run on wandb.
    :returns: str: The path to the saved best model.
    """


```




## Contributing
Contributions are welcome! If you encounter any issues or have suggestions for improvement, please create a new issue or submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
The sentence-transformers library and HugginFace for providing pre-trained NLP models
The faiss library for efficient similarity search
The sklearn and networkx libraries for clustering and graph operations
OpenAI for providing language model embeddings


## Roadmap 
We will continue to come up with more feature rich updates and introduce more modalities like images using support for vision and multimodal models.