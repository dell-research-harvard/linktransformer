# Change Log

## 0.1.0 Beta Release
## 0.1.1 Feature Updates and bug fixes
    + Bug fixes 
    + Allowed dataframe to be used as input to training instead of just paths
    + Doc fixes
    + Added an inference function two calculate cosine similarity of pairs using a model
    + splits validation into test and also logs the test accuracy if test set is not specified; added test accuracy calculation at the end of training
    + Add training and eval on pairs of negatives - both in data processing and eval during training
    + Added option to supply pre-split datasets (into train-val-test)
    + Option to save val and test data before training
    + More API changes in the inference script
    + Added an inference function to cluster rows of a df using an LM
## 0.1.2 
    + Added inference function to evaluate all pairs of left and right keys to calculate distances
    + Added more robust model card generation and upload to huggingface hub
    
