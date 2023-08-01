import logging

from sentence_transformers.util import fullname

class ModelCardTemplate:
    __TAGS__ = ["link-transformers","sentence-transformers", "sentence-similarity"]
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
tags:
{TAGS}
{DATASETS}
---

# {MODEL_NAME}

At the core of it, this model is a sentence transformer model [sentence-transformers](https://www.SBERT.net) model- it just wraps around the class. 
It is designed for quick and easy record linkage through the LinkTransformer package (https://github.com/dell-research-harvard/linktransformer).
Notwithstanding that, it can be used for any sentence similarity task within the sentence-transformers framework as well. 
It maps sentences & paragraphs to a {NUM_DIMENSIONS} dimensional dense vector space and can be used for tasks like clustering or semantic search.
Take a look at the documentation of SBERT if you want to use this model for more than just record linkage.

{MODEL_DESCRIPTION}

## Usage (LinkTransformer)

Using this model becomes easy when you have [LinkTransformer](https://www.SBERT.net) installed:

```
pip install -U linktransformer
```

Then you can use the model like this:

```python
from linktransformer import LinkTransformer, lm_merge
import pandas as pd

##Load the two dataframes that you want to link. For example, 2 dataframes with company names that are written differently
df1=pd.read_csv("data/df1.csv") ###This is the left dataframe with key CompanyName for instance
df2=pd.read_csv("data/df2.csv") ###This is the right dataframe with key CompanyName for instance

###Merge the two dataframes on the key column!
df_merged = lm_merge_df(df1, df2, on="CompanyName", how="inner")

##Done! The merged dataframe has a column called "score" that contains the similarity score between the two company names

You can also use this package for deduplication (clusters a df on the supplied key column). Merging a fine class (like product) to a coarse class (like HS code) is also possible.
Read our paper and the documentation for more!

```

{USAGE_TRANSFORMERS_SECTION}

## Evaluation Results

{EVALUATION}

You can evaluate the model using the [LinkTransformer](https://github.com/dell-research-harvard/linktransformer) package's inference functions.

{TRAINING_SECTION}

## Full Model Architecture
```
{FULL_MODEL_STR}
```

## Citing & Authors

{CITING}

"""



    __TRAINING_SECTION__ = """
## Training
The model was trained using SupCon loss. 

Parameters of the fit()-Method:
```
{FIT_PARAMETERS}
```

"""


    __USAGE_TRANSFORMERS__ = """\n
## Usage (HuggingFace Transformers)

### Record Linkage/Dedup/Aggregation
Any HuggingFace transformer can be used as an LinkTransformer model. It will just add a mean pooling layer on top of it to get sentence embeddings.
Usage can be found in the package docs. 

### Training your own LinkTransformer model
Any Sentence Transformers can be used as a backbone by simply adding a pooling layer.  Any other transformer on HuggingFace can also be used by specifying the option add_pooling_layer==True

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

