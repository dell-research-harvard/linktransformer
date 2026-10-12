# Change Log

## 0.2.0
        Packaging
        - Dependencies are now version ranges instead of exact pins, tested in CI at both the
          oldest allowed and newest versions (numpy>=2.0, pandas>=2.2.2, torch>=2.6,
          sentence-transformers>=5.0, transformers>=4.56, openai>=1.55.3, ...).
        - Training and provider libraries moved to extras. `pip install linktransformer` covers
          inference; use `linktransformer[train]` for train_model/train_clf_model (datasets,
          accelerate), `[wandb]` for W&B logging, `[gemini]` for Gemini, or `[all]`.
          Missing optional libraries raise an ImportError naming the extra to install.
        - Removed unused or indirect dependencies: faiss-cpu, scipy, httpx, tiktoken, evaluate,
          google-generativeai. Removed the stale requirements.txt.
        - The package version is defined once, in linktransformer/__init__.py.

        Behaviour and performance
        - `import linktransformer` is now near-instant; modules load on first use. The public
          API (every `lt.<name>`) is unchanged.
        - Nearest-neighbour and range search run in torch instead of FAISS (exact search, same
          results). This fixes segfaults/"OMP: Error #15" aborts on macOS when FAISS and torch
          were used in one process, and runs on GPU when CUDA is available.
        - merge_knn caps k at the size of the right dataframe (FAISS previously returned
          invalid -1 indices that matched the last row).
        - classify_rows with a local classifier predicts with a plain batched forward pass
          (identical outputs) instead of a transformers Trainer.
        - Classifier training metrics are computed with scikit-learn directly (same values as
          the `evaluate` metrics, without downloading metric scripts).

        sentence-transformers 6 / transformers 5.19 compatibility
        - Fixed LinkTransformer silently loading a different model than SentenceTransformer on
          sentence-transformers >= 5.5 (dropped Normalize layers, rebuilt default pooling).
          Embeddings now match SentenceTransformer exactly on 5.x and 6.x.
        - LinkTransformer.save delegates to sentence-transformers (it crashed on >= 5.4) and
          records the checkpoint as a SentenceTransformer, so models saved by LinkTransformer load
          identically with plain sentence-transformers. Checkpoints saved as "LinkTransformer" by
          earlier versions still load.
        - Fixed the pair-classification evaluator used with onlinecontrastive training
          (KeyError: 'cossim' on sentence-transformers >= 5.4), moved to the new import paths
          with fallbacks, and pass warmup as a step count so `warm_up_perc` keeps its meaning.
        - train_clf_model maps `warmup_ratio` (removed in transformers 5.15) instead of silently
          ignoring it, and warns about other unsupported training arguments.

        Model cards
        - Rewritten LinkTransformer and classifier cards: working code examples (`lt.merge` with
          the model, `lt.train_model`, sentence-transformers and transformers usage), Hub metadata
          (library_name, base_model, language), the published ACL 2024 citation, a proper model
          architecture block, and no unfilled {PLACEHOLDERS}.
        - Trained models get a LinkTransformer card (previously a fine-tuned model inherited its
          base model's README); cards report the loss and settings from LT_training_config.json.
        - Classifier cards report the real base model (previously the architecture name, e.g.
          "roberta"); train_clf_model records it in LT_training_config.json.

        Provider APIs
        - Gemini uses the `google-genai` SDK (google-generativeai is deprecated). Embeddings
          are requested in batches of 100 instead of one call per string; the merge_k_judge
          Gemini judge requests JSON output.
        - OpenAI embeddings use a client instance instead of the module-level `openai.api_key`,
          and are batched without the previous off-by-one in the split (from #31, @julienberman).
        - OpenAI chat calls send `max_completion_tokens` and drop parameters a model rejects
          (e.g. temperature for reasoning models); classification no longer caps answers at one
          token, and answers like "Yes." map to labels. Only rate limits, timeouts and server
          errors are retried (a 400 used to sleep for minutes per row).

        Embedding cache
        - New `embedding_cache="folder"` argument on every function that embeds text and on
          infer_embeddings. Embeddings are stored per model and text in a SQLite file, so both
          sides of a merge and all functions share them and only new texts are embedded, e.g.
          matching new rows against a precomputed corpus (#18, @Rinderkm; follows the design
          discussion in #31 with @julienberman). Local models are keyed by a fingerprint of
          their weights, so a retrained model never reuses stale embeddings.

        Fixes and contributions
        - An explicit `sep_token` passed to serialize_columns is respected, and OpenAI/Gemini
          models use "<SEP>" without querying Hugging Face for a tokenizer (#27, @pranjalv123).
          Multi-column OpenAI classify_rows/transform_rows now join columns with a space as
          intended (previously "</s>").
        - classify_rows accepts `tokenizer=` and falls back to the base model's tokenizer for
          checkpoints saved without tokenizer files (#20, @oooyiyangc). Recent transformers
          versions otherwise build an empty tokenizer that gives every row the same prediction.
        - pandas 3 support: DataFrame.map instead of the removed applymap, and string dtypes in
          classifier preprocessing (#30, @julienberman); Read the Docs builds from the package.
        - aggregate_rows no longer passes the deprecated merge_type (#14, #16).
        - Probable fix for segmentation faults in lt.merge (#21), see the FAISS change above.
        - train_clf_model no longer writes a tokenizer into a folder named "None" when no
          checkpoint beats the base model.
        - cluster_rows/dedup_rows no longer mutate a shared default `cluster_params` dict.
        - load_model raises a clear error instead of an UnboundLocalError when loading fails.
        - Training modules no longer add the package directory to sys.path.
        - Informational output uses the `logging` module instead of print, and training
          modules no longer configure the root logger or transformers' verbosity at import.

## 0.1.18
        - Added `merge_k_judge` as the primary end-to-end record linkage workflow:
            1) retrieve top-k candidates with `merge_knn`, then
            2) adjudicate candidate pairs with an LLM returning match + confidence.
        - Added explicit retrieval and judge model controls for `merge_k_judge`:
            `knn_sbert_model`, `knn_api_model`, and required `judge_llm_model`.
        - Added judge provider routing (`llm_provider`) with support for both OpenAI and Gemini judge paths.
        - Added strict failure semantics for judged linkage: retrieval errors and judge errors now raise clear exceptions
            (with guidance to use `merge_knn` directly when judgement is not desired).
        - Improved model resolution transparency with explicit warnings describing retrieval source/model/mode
            and judge source/model/provider.
        - Improved API-embedding serialization in `merge` and `merge_knn`:
            multi-column joins now serialize with `<SEP>` and avoid unnecessary tokenizer/org probing logs.
        - Added and stabilized row transformation workflows (`transform_rows`) for prompt-based normalization
            over one or multiple columns.
        - Expanded inference coverage with live-key-gated and monkeypatched tests across:
            SBERT retrieval, OpenAI embeddings retrieval, OpenAI judge, Gemini judge, and mixed-provider workflows.
        - Removed migration/debug artifacts in package source:
            hardcoded local path defaults, hardcoded legacy OpenAI embedding model fallback,
            and script-style debug runners in library modules.

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
## 0.1.17
    - Fixed the bug caused by a version of httpx for the openai api-dependant functions. 



    
