# LinkTransformer

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![arXiv](https://img.shields.io/badge/arXiv-2309.00789-b31b1b.svg)](https://arxiv.org/abs/2309.00789)

![Linkktransformers demo](https://github.com/dell-research-harvard/linktransformer/assets/60428420/15162828-b0fb-4ee3-8a0f-fdf3371d10a0)


LinkTransformer is a Python library for merging and deduplicating data frames using language model embeddings. It leverages popular Sentence Transformer (or any HuggingFace) models to generate embeddings for text data and provides functions to perform efficient 1:1, 1:m, and m:1 merges based on the similarity of embeddings. Additionally, the package includes utilities for clustering and data preprocessing. It also includes modifications to Sentence Transformers that allow for logging training runs on weights and biases.

- [Paper](https://arxiv.org/abs/2309.00789)
- [Website](https://linktransformer.github.io/)
- [Demo Video](https://www.youtube.com/watch?v=Sn47nmCvV9M)
- Tutorials
  + [Link Records with LinkTransformer](https://colab.research.google.com/drive/1OqUB8sqpUvrnC8oa_1RoOUzV6DaAKL4N?usp=sharing)
  + [Train your own LinkTransformer Model](https://colab.research.google.com/drive/1tHitPGjMMI2Nvh4wwA8rdcbYfbLaJDvg?usp=sharing)
  + [Classify text with LinkTransformer](https://colab.research.google.com/drive/1hSh_p8j7LP2RfdtxrPslOfnogC_CbYw5?usp=sharing)
- [Feature Deck](https://www.dropbox.com/scl/fi/dquxru8bndlyf9na14cw6/A-python-package-to-do-easy-record-linkage-using-Transformer-models.pdf?rlkey=fiv7j6c0vgl901y940054eptk&dl=0)

More tutorials are coming soon!


## Features

- Merge dataframes using language model embeddings
- Deduplicate data based on similarity threshold
- Efficient 1:1, 1:m, and m:1 merges
- Clustering methods for grouping similar data
- Support for various NLP models available on HuggingFace
- Classification - prediction and training in one line of code!

## Coming soon
- FAISS GPU, cuDF, cuML and cuGraph integration
- Convenience wrapper to use our models (trained on UN products and Wikidata)
- Integration of other modalities in this framework (Vision/Multimodal models)
- Hard negative mining for efficient training


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

![Linkktransformers demo_example](https://github.com/dell-research-harvard/linktransformer/assets/60428420/9f9fb9b8-a82b-4f2f-a111-4ee99baa0818)


```python
def merge(df1, df2, merge_type='1:1', on=None, model='your-pretrained-model', left_on=None, right_on=None, suffixes=('_x', '_y'),
                 use_gpu=False, batch_size=128, pooling_type='mean', openai_key=None):
    """
    Merge two dataframes using language model embeddings.

    :param df1 (DataFrame): First dataframe (left).
    :param df2 (DataFrame): Second dataframe (right).
    :param merge_type (str): Type of merge to perform (1:m or m:1 or 1:1).
    :param model (str): Language model to use.
    :param on (Union[str, List[str]], optional): Column(s) to join on in df1. Defaults to None.
    :param left_on (Union[str, List[str]], optional): Column(s) to join on in df1. Defaults to None.
    :param right_on (Union[str, List[str]], optional): Column(s) to join on in df2. Defaults to None.
    :param suffixes (Tuple[str, str]): Suffixes to use for overlapping columns. Defaults to ('_x', '_y').
    :param use_gpu (bool): Whether to use GPU. Not supported yet. Defaults to False.
    :param batch_size (int): Batch size for inferencing embeddings. Defaults to 128.
    :param openai_key (str, optional): OpenAI API key for InferKit API. Defaults to None.
    :return: DataFrame: The merged dataframe.
    """


```

A special case of merging is aggregation (use function: aggregate_rows)- when the left key is a list of items that need aggregation to the right keys. Semantic linking is also allowed with multiple columns as keys in both datasets. For larger datasets, merge_blocking can be used to merge within blocking keys. 


### Clustering or Deduplicating Data

![Linkktransformers demo_dedup](https://github.com/dell-research-harvard/linktransformer/assets/60428420/91c71fd3-dfe4-4918-bf4e-61f9189a35ff)


```python
def dedup_rows(df, model, on, threshold=0.5, openai_key=None):
    """
    Deduplicate a dataframe based on a similarity threshold. This is just clustering and keeping the first row in each cluster.
    Refer to the docs for the cluster_rows function for more details.

    :param df (DataFrame): Dataframe to deduplicate.
    :param model (str): Language model to use.
    :param on (Union[str, List[str]]): Column(s) to deduplicate on.
    :param cluster_type (str): Clustering method to use. Defaults to "SLINK".
    :param cluster_params (Dict[str, Any]): Parameters for clustering method. Defaults to {'threshold': 0.5, "min cluster size": 2, "metric": "cosine"}.
    :param openai_key (str): OpenAI API key
    :return: DataFrame: The deduplicated dataframe.
    """

def cluster_rows(df, model, on, threshold=0.5, openai_key=None):
    """
    Deduplicate a dataframe based on a similarity threshold. Various clustering options are supported.         
    "agglomerative": {
            "threshold": 0.5,
            "clustering linkage": "ward",  # You can choose a default linkage method
            "metric": "euclidean",  # You can choose a default metric
        },
        "HDBScan": {
            "min cluster size": 5,
            "min samples": 1,
        },
        "SLINK": {
            "min cluster size": 2,
            "threshold": 0.1,
        },
    }

    :param df (DataFrame): Dataframe to deduplicate.
    :param model (str): Language model to use.
    :param on (Union[str, List[str]]): Column(s) to deduplicate on.
    :param cluster_type (str): Clustering method to use. Defaults to "SLINK".
    :param cluster_params (Dict[str, Any]): Parameters for clustering method. Defaults to {'threshold': 0.5, "min cluster size": 2, "metric": "cosine"}.
    :param openai_key (str): OpenAI API key
    :return: DataFrame: The deduplicated dataframe.
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
    data: Union[str, pd.DataFrame] = None,
    train_data: Union[str, pd.DataFrame] = None,
    val_data: Union[str, pd.DataFrame] = None,
    test_data: Union[str, pd.DataFrame] = None,
    model_path: str="sentence-transformers/paraphrase-xlm-r-multilingual-v1",
    left_col_names: List[str] = ["description47"],
    right_col_names: List[str] = ['description48'],
    left_id_name: List[str] = ['tariffcode47'],
    right_id_name: List[str] = ['tariffcode48'],
    label_col_name: str = None,
    config_path: str = LINKAGE_CONFIG_PATH,
    training_args: dict = {"num_epochs":10},
    log_wandb: bool = False,
) -> str:
    """
    Train the LinkTransformer model.

    :param: model_path (str): The name of the model to use.
    :param: data (str): Path to the dataset in Excel or CSV format or a dataframe object.
    :param: left_col_names (List[str]): List of column names to use as left side data.
    :param: right_col_names (List[str]): List of column names to use as right side data.
    :param: left_id_name (List[str]): List of column names to use as identifiers for the left data.
    :param: right_id_name (List[str]): List of column names to use as identifiers for the right data,
    :param: label_col_name (str): Name of the column to use as labels. Specify this if you have data of the form (left, right, label). This type supports both positive and negative examples.
    :param: clusterid_col_name (str): Name of the column to use as cluster ids. Specify this if you have data of the form (text, cluster_id). 
    :param: cluster_text_col_name (str): Name of the column to use as cluster text. Specify this if you have data of the form (text, cluster_id).
    :param: config_path (str): Path to the JSON configuration file.
    :param: training_args (dict): Dictionary of training arguments to override the config.
    :param: log_wandb (bool): Whether to log the training run on wandb.
    :return: The path to the saved best model.
    """

```

### You can even Classify rows of text into predefined classes! 

#### Use pretrained models (ChatGPT or HuggingFace!)

```python

def classify_rows(
    df: DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    model: str = None,
    num_labels: int = 2,
    label_map: Optional[dict] = None,
    use_gpu: bool = False,
    batch_size: int = 128,
    openai_key: Optional[str] = None,
    openai_topic: Optional[str] = None,
    openai_prompt: Optional[str] = None,
    openai_params: Optional[dict] = {}
):
    """
    Classify texts in all rows of one or more columns whether they are relevant to a certain topic. The function uses
    either a trained classifier to make predictions or an OpenAI API key to send requests and retrieve classification
    results from ChatCompletion endpoint. The function returns a copy of the input dataframe with a new column "clf_preds_{on}" that stores the
    classification results.

    :param df: (DataFrame) the dataframe.
    :param on: (Union[str, List[str]], optional) Column(s) to classify (if multiple columns are passed in, they will be joined).
    :param model: (str) filepath to the model to use (to use OpenAI, see "https://platform.openai.com/docs/models").
    :param num_labels: (int) number of labels to predict. Defaults to 2.
    :param label_map: (dict) a dictionary that maps text labels to numeric labels. Used for OpenAI predictions.
    :param use_gpu: (bool) Whether to use GPU. Not supported yet. Defaults to False.
    :param batch_size: (int) Batch size for inferencing embeddings. Defaults to 128.
    :param openai_key: (str, optional) OpenAI API key for InferKit API. Defaults to None.
    :param openai_topic: (str, optional) The topic predict whether the text is relevant or not. Defaults to None.
    :param openai_prompt: (str, optional) Custom system prompt for OpenAI ChatCompletion endpoint. Defaults to None.
    :param openai_params: (str, optional) Custom parameters for OpenAI ChatCompletion endpoint. Defaults to None.
    :returns: DataFrame: The dataframe with a new column "clf_preds_{on}" that stores the classification results.
    """

```


#### Train your own model! 

```python
def train_clf_model(data=None,model="distilroberta-base",on=[],label_col_name="label",train_data=None,val_data=None,test_data=None,data_dir=".",
                    training_args={},config=CLF_CONFIG_PATH,
                    eval_steps=None,save_steps=None,batch_size=None,lr=None,
                    epochs=None,model_save_dir=".", weighted_loss=False,weight_list=None,
                    wandb_log=False,wandb_name="topic",
                    print_test_mistakes=False):
    """
    Trains a text classification model using Hugging Face's Transformers library.
    
    :param data: (str/DataFrame, optional) Path to the CSV file or a DataFrame object containing the training data.
    :param model: (str, default="distilroberta-base") The name of the Hugging Face model to be used.
    :param on: (list, default=[]) List of column names that are used as input features.
    :param label_col_name: (str, default="label") The column name in the data that contains the labels.
    :param train_data: (str/DataFrame, optional) Training dataset if `data` is not provided.
    :param val_data: (str/DataFrame, optional) Validation dataset if `data` is not provided.
    :param test_data: (str/DataFrame, optional) Test dataset if `data` is not provided.
    :param data_dir: (str, default=".") Directory where training data splits are saved.
    :param training_args: (dict, default={}) Training arguments for the Hugging Face Trainer.
    :param config: (str, default=CLF_CONFIG_PATH) Path to the default config file.
    :param eval_steps: (int, optional) Evaluation interval in terms of steps.
    :param save_steps: (int, optional) Model saving interval in terms of steps.
    :param batch_size: (int, optional) Batch size for training and evaluation.
    :param lr: (float, optional) Learning rate.
    :param epochs: (int, optional) Number of training epochs.
    :param model_save_dir: (str, default=".") Directory where the trained model will be saved.
    :param weighted_loss: (bool, default=False) If true, uses weighted loss based on class frequencies.
    :param weight_list: (list, optional) Weights for each class in the loss function.
    :param wandb_log: (bool, default=False) If true, logs metrics to Weights & Biases.
    :param wandb_name: (str, default="topic") Name of the Weights & Biases project.
    :param print_test_mistakes: (bool, default=False) If true, prints the misclassified samples in the test dataset.
    
    :return: 
        - best_model_path (str): Path to the directory of the best saved model.
        - best_metric (float): The best metric value achieved during training.
        - label_map (dict): Mapping of labels to their respective integer values.
        
    Note:
        Either the `data` parameter or all of `train_data`, `val_data`, and `test_data` should be provided. If only
        `data` is provided, it will be split into train, validation, and test sets.
    """


```


## Contributing
Contributions are welcome! If you encounter any issues or have suggestions for improvement, please create a new issue or submit a pull request.

## License
This project is licensed under the GNU General Public License- see the LICENSE file for details.

## Acknowledgments

- - The sentence-transformers library and HugginFace for providing pre-trained NLP models
- - The faiss library for efficient similarity search
- - The sklearn and networkx libraries for clustering and graph operations
- - OpenAI for providing language model embeddings


## Roadmap 
We will continue to come up with more feature-rich updates and introduce more modalities like images using support for vision and multimodal models within this framework to make those accessible to those with a non-technical background. 

## Package Maintainers
- Sam Jones (samuelcaronnajones)
- Abhishek Arora (econabhishek)
- Yiyang Chen (oooyiyangc)
