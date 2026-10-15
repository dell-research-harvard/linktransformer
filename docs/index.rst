LinkTransformer
===============

LinkTransformer links, deduplicates, clusters and classifies the rows of pandas DataFrames with
transformer language models: it embeds the key columns of each row and matches rows by cosine
similarity.

- Paper: `LinkTransformer, ACL 2024 System Demonstrations <https://aclanthology.org/2024.acl-demos.21/>`_
- Source code, tutorials and guides: `GitHub <https://github.com/dell-research-harvard/linktransformer>`_
- Website: https://linktransformer.github.io/

.. note::
   Version 0.1.19 is a testing release of the changes planned for 0.2.0. It changes how clustering
   thresholds work and moves training dependencies to an extra; see the
   `change log <https://github.com/dell-research-harvard/linktransformer/blob/main/CHANGELOG.md>`_.
   Please report problems on GitHub.

Installation
------------

.. code-block:: bash

   pip install linktransformer

Optional extras: ``train`` (fine-tuning with :func:`~linktransformer.train_model.train_model` and
:func:`~linktransformer.train_clf_model.train_clf_model`), ``wandb`` (Weights & Biases logging),
``gemini`` (Gemini models) and ``all``, e.g. ``pip install "linktransformer[train]"``.

Quick start
-----------

.. code-block:: python

   import pandas as pd
   import linktransformer as lt

   left = pd.DataFrame({"CompanyName": ["Tech Corporation", "Acme Inc"]})
   right = pd.DataFrame({"CompanyName": ["Tech Corp", "ACME Incorporated", "Globex"]})

   # the 2 most similar right rows for every left row, with a similarity score
   matches = lt.merge_knn(left, right, on="CompanyName", model="all-MiniLM-L6-v2", k=2)

   # group near-duplicate names: rows at least this similar belong together
   clusters = lt.cluster_rows(right, on="CompanyName", model="all-MiniLM-L6-v2", sim_threshold=0.85)

Main functions
--------------

- Link two dataframes: :func:`~linktransformer.infer.merge`, :func:`~linktransformer.infer.merge_knn`,
  :func:`~linktransformer.infer.merge_range`, :func:`~linktransformer.infer.merge_blocking`,
  :func:`~linktransformer.infer.aggregate_rows`
- Retrieve candidates, then verify them with an LLM: :func:`~linktransformer.infer.merge_k_judge`,
  :func:`~linktransformer.infer.judge_merge_k_out`
- Group or remove near-duplicates: :func:`~linktransformer.infer.cluster_rows`,
  :func:`~linktransformer.infer.dedup_rows`, :func:`~linktransformer.cluster_fns.cluster_embeddings`
- Check a merge's output: :func:`~linktransformer.diagnostics.match_report`
- Classify or rewrite rows: :func:`~linktransformer.infer.classify_rows`,
  :func:`~linktransformer.infer.transform_rows`
- Fine-tune models on your own data: :func:`~linktransformer.train_model.train_model`,
  :func:`~linktransformer.train_clf_model.train_clf_model`

Using with AI agents
--------------------

The repository includes an agent skill, `skills/linktransformer
<https://github.com/dell-research-harvard/linktransformer/tree/main/skills/linktransformer>`_, that teaches
coding agents to pick the function and model, check a similarity threshold, judge candidate pairs within a
budget and run long jobs resumably. Its ``SKILL.md`` lists the reference files and when to read each. The
`README <https://github.com/dell-research-harvard/linktransformer#using-with-ai-agents>`_ shows how to
install it as a Claude Code plugin.

.. toctree::
   :maxdepth: 2
   :caption: API reference

   modules
