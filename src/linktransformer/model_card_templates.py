
class ModelCardTemplate:
    __TAGS__ = ["linktransformer","sentence-transformers", "sentence-similarity","tabular-classification"]
    __DEFAULT_VARS__ = {
        "{PIPELINE_TAG}": "sentence-similarity",
        "{MODEL_DESCRIPTION}": "<!--- Describe your model here -->",
        "{TRAINING_SECTION}": "",
        "{USAGE_TRANSFORMERS_SECTION}": "",
        "{EVALUATION}": "<!--- Describe how your model was evaluated -->",
        "{CITING}": "<!--- Describe where people can find more information -->"
    }

    __MODEL_CARD__ = """
---
pipeline_tag: {PIPELINE_TAG}
language: 
{LANGUAGE}
tags:
{TAGS}
{DATASETS}
---

# {MODEL_NAME}

This is a [LinkTransformer](https://linktransformer.github.io/) model. At its core this model this is a sentence transformer model [sentence-transformers](https://www.SBERT.net) model- it just wraps around the class. 
It is designed for quick and easy record linkage (entity-matching) through the LinkTransformer package. The tasks include clustering, deduplication, linking, aggregation and more.
Notwithstanding that, it can be used for any sentence similarity task within the sentence-transformers framework as well. 
It maps sentences & paragraphs to a {NUM_DIMENSIONS} dimensional dense vector space and can be used for tasks like clustering or semantic search.
Take a look at the documentation of [sentence-transformers](https://www.sbert.net/index.html) if you want to use this model for more than what we support in our applications. 


This model has been fine-tuned on the model : {BASE_MODEL}. It is pretrained for the language : {LANGUAGE}.


{MODEL_DESCRIPTION}

## Usage (LinkTransformer)

Using this model becomes easy when you have [LinkTransformer](https://github.com/dell-research-harvard/linktransformer) installed:

```
pip install -U linktransformer
```

Then you can use the model like this:

```python
import linktransformer as lt
import pandas as pd

##Load the two dataframes that you want to link. For example, 2 dataframes with company names that are written differently
df1=pd.read_csv("data/df1.csv") ###This is the left dataframe with key CompanyName for instance
df2=pd.read_csv("data/df2.csv") ###This is the right dataframe with key CompanyName for instance

###Merge the two dataframes on the key column!
df_merged = lt.merge(df1, df2, on="CompanyName", how="inner")

##Done! The merged dataframe has a column called "score" that contains the similarity score between the two company names

```


## Training your own LinkTransformer model
Any Sentence Transformers can be used as a backbone by simply adding a pooling layer.  Any other transformer on HuggingFace can also be used by specifying the option add_pooling_layer==True
The model was trained using SupCon loss. 
Usage can be found in the package docs. 
The training config can be found in the repo with the name LT_training_config.json
To replicate the training, you can download the file and specify the path in the config_path argument of the training function. You can also override the config by specifying the training_args argument.
Here is an example. 


```python

##Consider the example in the paper that has a dataset of Mexican products and their tariff codes from 1947 and 1948 and we want train a model to link the two tariff codes.
saved_model_path = train_model(
        model_path="hiiamsid/sentence_similarity_spanish_es",
        dataset_path=dataset_path,
        left_col_names=["description47"],
        right_col_names=['description48'],
        left_id_name=['tariffcode47'],
        right_id_name=['tariffcode48'],
        log_wandb=False,
        config_path=LINKAGE_CONFIG_PATH,
        training_args={"num_epochs": 1}
    )

```


You can also use this package for deduplication (clusters a df on the supplied key column). Merging a fine class (like product) to a coarse class (like HS code) is also possible.
Read our paper and the documentation for more!

{USAGE_TRANSFORMERS_SECTION}

## Evaluation Results

{EVALUATION}

You can evaluate the model using the [LinkTransformer](https://github.com/dell-research-harvard/linktransformer) package's inference functions.
We have provided a few datasets in the package for you to try out. We plan to host more datasets on Huggingface and our website (Coming soon) that you can take a look at.

{TRAINING_SECTION}



{FULL_MODEL_STR}
```

## Citing & Authors

```
@misc{arora2023linktransformer,
                  title={LinkTransformer: A Unified Package for Record Linkage with Transformer Language Models},
                  author={Abhishek Arora and Melissa Dell},
                  year={2023},
                  eprint={2309.00789},
                  archivePrefix={arXiv},
                  primaryClass={cs.CL}
                }

```

"""



    __TRAINING_SECTION__ = """
## Training

### Training your own LinkTransformer model
Any Sentence Transformers can be used as a backbone by simply adding a pooling layer.  Any other transformer on HuggingFace can also be used by specifying the option add_pooling_layer==True
The model was trained using SupCon loss. 
Usage can be found in the package docs. 
The training config can be found in the repo with the name LT_training_config.json
To replicate the training, you can download the file and specify the path in the config_path argument of the training function. You can also override the config by specifying the training_args argument.
Here is an example. 


```python

##Consider the example in the paper that has a dataset of Mexican products and their tariff codes from 1947 and 1948 and we want train a model to link the two tariff codes.
saved_model_path = train_model(
        model_path="hiiamsid/sentence_similarity_spanish_es",
        dataset_path=dataset_path,
        left_col_names=["description47"],
        right_col_names=['description48'],
        left_id_name=['tariffcode47'],
        right_id_name=['tariffcode48'],
        log_wandb=False,
        config_path=LINKAGE_CONFIG_PATH,
        training_args={"num_epochs": 1}
    )


```



"""


class ClassificationModelCardTemplate:
    __TAGS__ = ["linktransformer", "transformers", "text-classification", "tabular-classification"]
    __DEFAULT_VARS__ = {
        "{PIPELINE_TAG}": "text-classification",
        "{MODEL_DESCRIPTION}": "<!--- Describe your model here -->",
        "{TRAINING_SECTION}": "",
        "{USAGE_TRANSFORMERS_SECTION}": "",
        "{EVALUATION}": "<!--- Describe how your model was evaluated -->",
        "{CITING}": "<!--- Describe where people can find more information -->"
    }

    __MODEL_CARD__ = """
---
pipeline_tag: {PIPELINE_TAG}
language: 
{LANGUAGE}
tags:
{TAGS}
{DATASETS}
---

# {MODEL_NAME}

This model is part of the [LinkTransformer](https://linktransformer.github.io/) ecosystem. While rooted in the a standard HuggingFace Transformer, this specific instance is tailored for text classification tasks. It classifies input sentences or paragraphs into specific categories or labels, leveraging the power of transformer architectures.

The base model for this classifier is: {BASE_MODEL}. It is pretrained for the language: {LANGUAGE}.

Labels are mapped to integers as follows: 

{LABEL_MAP}

{MODEL_DESCRIPTION}

## Usage with LinkTransformer

After installing [LinkTransformer](https://linktransformer.github.io/):

```python
pip install -U linktransformer
```

Employ the model for text classification tasks:

```python
import linktransformer as lt
df_clf_output = lt.classify_rows(df, on=["col_of_interest"], model="{MODEL_NAME}")
```


## Training

### Training your own LinkTransformer Classification Model

With the provided tools, you can train a custom classification model:

```python
from linktransformer import train_clf_model

best_model_path, best_metric, label_map = train_clf_model(
    data="path_to_dataset.csv",
    model="you-model-path-or-name",
    on=["col_of_interest"],
    label_col_name="label_column_name",
    lr=5e-5,
    batch_size=16,
    epochs=3
)
```

{TRAINING_SECTION}

{USAGE_TRANSFORMERS_SECTION}

## Evaluation Results

{EVALUATION}

Evaluation is typically based on metrics like accuracy, F1-score, precision, and recall.

## Citing & Authors

```
@misc{arora2023linktransformer,
                  title={LinkTransformer: A Unified Package for Record Linkage with Transformer Language Models},
                  author={Abhishek Arora and Melissa Dell},
                  year={2023},
                  eprint={2309.00789},
                  archivePrefix={arXiv},
                  primaryClass={cs.CL}
}
```

"""

    __TRAINING_SECTION__ = """
## Training

### Training your own LinkTransformer Classification Model

With the provided tools, you can train a custom classification model:

```python
from linktransformer import train_clf_model

best_model_path, best_metric, label_map = train_clf_model(
    data="path_to_dataset.csv",
    on=["text_column_name"],
    label_col_name="label_column_name",
    lr=5e-5,
    batch_size=16,
    epochs=3
)
```

More information on configurations can be found in the [LinkTransformer documentation](https://linktransformer.github.io/).
"""

