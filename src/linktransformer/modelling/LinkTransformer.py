###Modifications to the Base SentenceTransformers class. We need to re-write the saving functions to make LinkTransformer models out of them.
import logging
from sentence_transformers import SentenceTransformer
from typing import Optional, List, Iterable, Union
from huggingface_hub import HfApi, get_token
import os
import json
from linktransformer.model_card_templates import render_linktransformer_card
from linktransformer.progress import hf_progress_bars_off_if_disabled
import shutil
import stat
from torch import nn
import tempfile





logger = logging.getLogger(__name__)

# Tokenizers that transformers 5 does not handle correctly: 5.0 and 5.1 load them but produce wrong token ids
# (silently wrong embeddings), later versions fail to load them. Checked up to transformers 5.19.
_TOKENIZERS_BROKEN_IN_TRANSFORMERS_5 = {"MLukeTokenizer"}


def _transformers_is_v5_or_later() -> bool:
    import transformers
    from packaging.version import Version

    return Version(transformers.__version__).major >= 5


def _declared_tokenizer_class(model_name_or_path: str) -> Optional[str]:
    """The tokenizer class a checkpoint declares in tokenizer_config.json, read without loading the tokenizer."""
    try:
        if os.path.isdir(model_name_or_path):
            config_path = os.path.join(model_name_or_path, "tokenizer_config.json")
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(model_name_or_path, "tokenizer_config.json")
        with open(config_path, encoding="utf8") as fIn:
            return json.load(fIn).get("tokenizer_class")
    except Exception:
        return None


def _unsupported_tokenizer_error(model_name_or_path: Optional[str], tokenizer_class: str) -> RuntimeError:
    import transformers

    return RuntimeError(
        f"{model_name_or_path!r} uses the {tokenizer_class}, which transformers {transformers.__version__} does not "
        "handle correctly (transformers 5.0-5.1 produce wrong token ids; later versions fail to load it). "
        "Use transformers 4.x for this model: pip install 'transformers>=4.56,<5' 'sentence-transformers>=5,<6'"
    )


def _copy_dir_contents(src_dir: str, dst_dir: str):
    for name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, name)
        dst_path = os.path.join(dst_dir, name)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)

class LinkTransformer(SentenceTransformer):
    """
    Modified SentenceTransformer class for LinkTransformers models as a wrapper around the SentenceTransformer class.

    Loads or create a LinkTransformers model, that can be used to map sentences / text to embeddings which in-turn can be used for LM based record linkage.

    :param model_name_or_path: If it is a filepath on disc, it loads the model from that path. If it is not a path, it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model from Huggingface models repository with that name.
    :param modules: This parameter can be used to create custom SentenceTransformer models from scratch.
    :param device: Device (like 'cuda' / 'cpu') that should be used for computation. If None, checks if a GPU can be used.
    :param cache_folder: Path to store models. Can be also set by SENTENCE_TRANSFORMERS_HOME enviroment variable.
    :param use_auth_token: HuggingFace authentication token to download private models.
    """
    def __init__(self, model_name_or_path: Optional[str] = None,
                 modules: Optional[Iterable[nn.Module]] = None,
                 device: Optional[str] = None,
                 cache_folder: Optional[str] = None,
                 use_auth_token: Union[bool, str, None] = None,
                 opt_model_description: Optional[str] = None,
                 opt_model_lang: Optional[str] = None,
                 ):


        try:
            with hf_progress_bars_off_if_disabled():
                super().__init__(model_name_or_path=model_name_or_path, modules=modules, device=device, cache_folder=cache_folder, token=use_auth_token)
        except Exception as exc:
            if model_name_or_path is not None and _transformers_is_v5_or_later():
                declared = _declared_tokenizer_class(model_name_or_path)
                if declared in _TOKENIZERS_BROKEN_IN_TRANSFORMERS_5:
                    raise _unsupported_tokenizer_error(model_name_or_path, declared) from exc
            raise
        loaded_tokenizer_class = type(getattr(self, "tokenizer", None)).__name__
        if loaded_tokenizer_class in _TOKENIZERS_BROKEN_IN_TRANSFORMERS_5 and _transformers_is_v5_or_later():
            raise _unsupported_tokenizer_error(model_name_or_path, loaded_tokenizer_class)

        self.base_model_name_or_path = model_name_or_path
        self.opt_model_description = opt_model_description
        self.opt_model_lang = opt_model_lang

        ##A locally saved LinkTransformer model carries its training config; use it for what was not passed explicitly
        config_path = os.path.join(model_name_or_path, 'LT_training_config.json') if model_name_or_path is not None else None
        if config_path is not None and os.path.isfile(config_path):
            with open(config_path, 'r') as fIn:
                self._lt_model_config = json.load(fIn)
            logger.info("Loaded LinkTransformer model config from {}".format(config_path))
            self.base_model_name_or_path = self._lt_model_config.get('base_model_path', model_name_or_path)
            if opt_model_description is None:
                self.opt_model_description = self._lt_model_config.get('opt_model_description')
            if opt_model_lang is None:
                self.opt_model_lang = self._lt_model_config.get('opt_model_lang')
            
    def _expected_model_type(self) -> str:
        """The model type sentence-transformers compares a checkpoint against when loading it into this class."""
        # sentence-transformers < 5.4 compares against _model_config["model_type"] (the class name);
        # later versions compare against the `model_type` class attribute ("SentenceTransformer").
        model_config = getattr(self, "_model_config", None)
        if isinstance(model_config, dict) and "model_type" in model_config:
            return model_config["model_type"]
        return getattr(type(self), "model_type", type(self).__name__)

    def _get_model_type(
        self,
        model_name_or_path: str,
        token: bool | str | None,
        cache_folder: str | None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> str | None:
        """
        Report SentenceTransformer and LinkTransformer checkpoints as this class's own model type, so
        sentence-transformers loads their saved modules as-is (see sentence-transformers#3536).

        Any other mismatch makes sentence-transformers rebuild the model with default modules (for
        example dropping a Normalize layer or changing the pooling), which silently changes embeddings.
        """
        stored_type = super()._get_model_type(
            model_name_or_path,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        if stored_type in (None, "SentenceTransformer", "LinkTransformer"):
            return self._expected_model_type()
        return stored_type


    def save(self, path: str, model_name: Optional[str] = None, create_model_card: bool = True, train_datasets: Optional[List[str]] = None,override_model_description: Optional[str] = None, 
                    override_model_lang: Optional[str] = None, safe_serialization: bool = True, **kwargs):
        """
        Saves all elements for this seq. sentence embedder into different sub-folders
        
        :param path: Path on disc
        :param model_name: Optional model name
        :param create_model_card: If True, create a README.md with basic information about this model
        :param train_datasets: Optional list with the names of the datasets used to to train the model
        :param override_model_description: Optional model description to override the default model description in the model card
        :param override_model_lang: Optional model language to override the default model language in the model card
        :param safe_serialization: Accepted for compatibility with modern SentenceTransformer save_pretrained API.
        :param kwargs: Additional keyword arguments accepted for forward compatibility.

        :return: None
        """
        if path is None:
            return
        
        if override_model_description is not None:
            self.opt_model_description = override_model_description
        if override_model_lang is not None:
            self.opt_model_lang = override_model_lang

        # Record the checkpoint as a SentenceTransformer so plain sentence-transformers loads it without
        # conversion (sentence-transformers < 5.4 would otherwise write "LinkTransformer" here).
        model_config = getattr(self, "_model_config", None)
        own_model_type = model_config.get("model_type") if isinstance(model_config, dict) else None
        if own_model_type is not None:
            model_config["model_type"] = "SentenceTransformer"
        try:
            # sentence-transformers writes the modules and configs in its current format and calls
            # self._create_model_card, so the LinkTransformer model card is kept.
            super().save(
                path,
                model_name=model_name,
                create_model_card=create_model_card,
                train_datasets=train_datasets,
                safe_serialization=safe_serialization,
            )
        finally:
            if own_model_type is not None:
                model_config["model_type"] = own_model_type

    def _create_model_card(self, path: str, model_name: Optional[str] = None, train_datasets: Optional[List[str]] = None):
        """
        Write the model card (README.md) to path.

        As in sentence-transformers, a card loaded from the Hub is kept as is unless the model has been
        trained since; trained models get a LinkTransformer card describing the base model and training.
        """
        card_data = getattr(self, "model_card_data", None)
        trained = "generated_from_trainer" in (getattr(card_data, "tags", None) or [])
        if self._model_card_text and not trained:
            model_card = self._model_card_text
        else:
            # train_model writes LT_training_config.json into the save directory before training
            training_config = getattr(self, "_lt_model_config", None)
            config_path = os.path.join(path, "LT_training_config.json")
            if os.path.isfile(config_path):
                with open(config_path, encoding="utf8") as fIn:
                    training_config = json.load(fIn)
            training_config = training_config or {}

            # get_embedding_dimension replaces get_sentence_embedding_dimension in sentence-transformers >= 5.4
            get_dimension = getattr(self, "get_embedding_dimension", None) or self.get_sentence_embedding_dimension
            model_card = render_linktransformer_card(
                model_name=(model_name or path).strip(),
                num_dimensions=get_dimension(),
                architecture=str(self),
                base_model=training_config.get("base_model_path") or getattr(self, "base_model_name_or_path", None),
                languages=self.opt_model_lang,
                description=self.opt_model_description,
                train_datasets=train_datasets if isinstance(train_datasets, (list, tuple)) else None,
                training_config=training_config,
            )

        with open(os.path.join(path, "README.md"), "w", encoding='utf8') as fOut:
            fOut.write(model_card)

    def save_to_hub(self,
                    repo_name: str,
                    organization: Optional[str] = None,
                    private: Optional[bool] = None,
                    commit_message: str = "Add new LinkTransformer model.",
                    local_model_path: Optional[str] = None,
                    exist_ok: bool = False,
                    replace_model_card: bool = False,
                    train_datasets: Optional[List[str]] = None,
                    replace_all_contents: bool = False,
                    retain_model_card: bool = True,
                    ):
        """
        Uploads all elements of this LinkTransformer (inherited Sentence Transformer) to a new HuggingFace Hub repository.

        :param repo_name: Repository name for your model in the Hub.
        :param organization:  Organization in which you want to push your model or tokenizer (you must be a member of this organization).
        :param private: Set to true, for hosting a prive model
        :param commit_message: Message to commit while pushing.
        :param local_model_path: Path of the model locally. If set, this file path will be uploaded. Otherwise, the current model will be uploaded
        :param exist_ok: If true, saving to an existing repository is OK. If false, saving only to a new repository is possible
        :param replace_model_card: If true, replace an existing model card in the hub with the automatically created model card
        :param train_datasets: Datasets used to train the model. If set, the datasets will be added to the model card in the Hub.
        :param replace_all_contents: If true, replace all contents of the repository with the current model. If false, only the model files will be replaced.
        :param retain_model_card: If true, the model card will be retained in the repository. If false, the model card will be deleted.
        :return: The url of the commit of your model in the given repository.
        """
        token = get_token() or os.getenv("HF_TOKEN")
        if token is None:
            raise ValueError("You must login to the Hugging Face hub. Run `hf auth login` or set `HF_TOKEN`.")

        if '/' in repo_name:
            splits = repo_name.split('/', maxsplit=1)
            if organization is None or organization == splits[0]:
                organization = splits[0]
                repo_name = splits[1]
            else:
                raise ValueError("You passed and invalid repository name: {}.".format(repo_name))

        api = HfApi()
        repo_id = repo_name
        if organization:
          repo_id = f"{organization}/{repo_id}"
        api.create_repo(
                repo_id=repo_id,
                token=token,
                private=private,
                repo_type=None,
                exist_ok=exist_ok,
            )
        full_model_name = repo_id

        with tempfile.TemporaryDirectory() as tmp_dir:
            if replace_all_contents:
                existing_files = api.list_repo_files(repo_id=repo_id, repo_type="model", token=token)
                for path_in_repo in existing_files:
                    if retain_model_card and path_in_repo == "README.md":
                        continue
                    try:
                        api.delete_file(
                            path_in_repo=path_in_repo,
                            repo_id=repo_id,
                            repo_type="model",
                            token=token,
                            commit_message="Clear repository contents",
                        )
                    except Exception as exc:
                        logger.warning("Failed deleting %s from repo: %s", path_in_repo, exc)
            
            
            # If user provides local files, copy them.
            if local_model_path:
                _copy_dir_contents(local_model_path, tmp_dir)
            else:  # Else, save model directly into local repo.
                create_model_card = replace_model_card or not os.path.exists(os.path.join(tmp_dir, 'README.md'))
                self.save(tmp_dir, model_name=full_model_name, create_model_card=create_model_card, train_datasets=train_datasets)

            ##Save the model config (_lt_model_config) to the repo if it exists
            if hasattr(self, '_lt_model_config'):
                with open(os.path.join(tmp_dir, 'LT_training_config.json'), 'w') as fOut:
                    json.dump(self._lt_model_config, fOut, indent=2)

            logger.info("Push model to the hub. This might take a while")
            push_return = api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=tmp_dir,
                token=token,
                commit_message=commit_message,
            )

            def on_rm_error(func, path, exc_info):
                # path contains the path of the file that couldn't be removed
                # let's just assume that it's read-only and unlink it.
                try:
                    os.chmod(path, stat.S_IWRITE)
                    os.unlink(path)
                except OSError:
                    pass

            try:
                for f in os.listdir(tmp_dir):
                    path = os.path.join(tmp_dir, f)
                    if os.path.isdir(path):
                        shutil.rmtree(path, onerror=on_rm_error)
                    else:
                        os.remove(path)
            except Exception as e:
                logger.warning("Error when deleting temp folder: {}".format(str(e)))
                pass


        return push_return
