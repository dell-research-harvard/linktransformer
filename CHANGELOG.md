# Change Log

## 0.1.0 Beta Release
## 0.1.1 Feature Updates and bug fixes
    - Bug fixes 
    - Allowed dataframe to be used as input to training instead of just paths
    - Doc fixes
    - Added an inference function two calculate cosine similarity of pairs using a model
    - splits validation into test and also logs the test accuracy if test set is not specified; added test accuracy calculation at the end of training
    - Add training and eval on pairs of negatives - both in data processing and eval during training
    - Added option to supply pre-split datasets (into train-val-test)
    - Option to save val and test data before training
    - More API changes in the inference script
    - Added an inference function to cluster rows of a df using an LM
## 0.1.2 
    - Added inference function to evaluate all pairs of left and right keys to calculate distances
    - Added more robust model card generation and upload to huggingface hub
## 0.1.3
    - Bug fixes
## 0.1.4
    - Bug fixes in training pipeline
    - Added a function to do merge k nearest neighbours instead of just one
    - Bug fixes in model card generation and upload to hugging face hub
    - Preprocessing data for training now gracefully handles cases where an id is not specified for left or right columns. It now groups by the two columns to handle exact duplicates. Id is still recommended. 
## 0.1.5
    - Guardrails added to preprocessing before training to group by keys in case column id is not specified instead of consdiering every pair unique
## 0.1.6 
    - Bug fixes in model uploads to the hub (missing training config)
## 0.1.7
    - Default param changes in train model function
    - Added splitting string arrays before feeding to OpenAI embeddings API to account for token limits
    - Doc changes
    - bug fixes in the function evaluate_pairs
    - Added an option to train on a dataset of cluster text and labels
## 0.1.8 
    - More fixes; performance improvements
    - Added classification features - infer using HF transformers, OpenAI (Chat), and train custom transformer models in 1 line
    - More demo datasets
## 0.1.9 
    - Minor bug fixes
## 0.1.10
    - Allowed a progress bar for classification inference
## 0.1.11 
    - Allowed a progress bar for training inference 
    - Fixed a bug in tokenizer saving 
## 0.1.12
    - Fixes in merge and classification inference to be in line with OpenAI API changes
    - Bug fixes in column serialisation when using open ai embeddings 
    - Updated toml file to restrict by package versioning - forward compatibility would not be supported
    - Made merge with blocking faster - no longer needs loading the model for each block
    - All Linkage inference functions can now take in a LinkTransformer model (wrapper around SentenceTransformer) as input. This would be useful for workflows requiring looping; preventing repeated model loading.
    - More robust typing where it was lacking
    - Fixes in cluster functions and bringing them to the root directory
## 0.1.13
    - Major Update : Allowed Online contrastive loss in model training with paired data with labels
    - Fixed the behaviour of the custom suffix feature in merge functions
    - Fixed an incorrect piece of preprocessing code for paired data with labels - we recommend training such models again for a substantial increase in performance
    - Allowed loss_type in the train_model args - "supcon" and "onlinecontrastive"
    - Made some changes in the linkage configs to allow loss params into the training args. loss_params is a dictionary containing "temperature" for supcon loss and "margin" for onlinecontrastive.
## 0.1.14 
    - Bug Fixes
## 0.1.15
    - Static dependency versioning - sentence-transformers underwent a massive update, until we are able to catch up, we will default to previously supported versions. 
## 0.1.16
    - Minor bug fixes due to dependency changes. Fixed issue that breaks installation on google colab.
    - Updated code for newest stable releases of transformers, pandas and sentence-transformers



    
