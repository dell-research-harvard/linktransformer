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
