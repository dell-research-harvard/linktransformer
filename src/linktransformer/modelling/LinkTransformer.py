###Modifications to the Base SentenceTransformers class. We need to re-write the saving functions to make LinkTransformer models out of them.
import logging
from sentence_transformers import SentenceTransformer
from typing import Optional, List, Iterable, Union
from huggingface_hub import HfApi, get_token
import transformers
from sentence_transformers import __version__ 
import os
import torch
import json
from linktransformer.model_card_templates import ModelCardTemplate
import shutil
import stat
from sentence_transformers.models import Transformer
from torch import nn
from linktransformer import __MODEL_HUB_ORGANIZATION__
import tempfile





logger = logging.getLogger(__name__)


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


        super().__init__(model_name_or_path=model_name_or_path, modules=modules, device=device, cache_folder=cache_folder, use_auth_token=use_auth_token)

        ##If it is a local path, we need to load the config (LT) from the file FLAG - CLEAN THIS UP
        if os.path.isdir(model_name_or_path):
            ###If config file exists, load it. It will be in the parent folder of the model
            if os.path.isfile(os.path.join(model_name_or_path, 'LT_training_config.json')):
                with open(os.path.join(model_name_or_path, 'LT_training_config.json'), 'r') as fIn:
                    self._lt_model_config = json.load(fIn)
                    print("Loaded LinkTransformer model config from {}".format(os.path.join(model_name_or_path, 'LT_training_config.json')))
                self.base_model_name_or_path = self._lt_model_config['base_model_path']
                self.opt_model_description = self._lt_model_config['opt_model_description']
                self.opt_model_lang = self._lt_model_config['opt_model_lang']
        else:
            ###If huggningface model, then assign ##Implement later
            self.base_model_name_or_path = model_name_or_path

            self.opt_model_description = opt_model_description
            self.opt_model_lang = opt_model_lang
            



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

        os.makedirs(path, exist_ok=True)

        logger.info("Save model to {}".format(path))
        modules_config = []

        #Save some model info
        if '__version__' not in self._model_config:
            self._model_config['__version__'] = {
                    'sentence_transformers': __version__,
                    'transformers': transformers.__version__,
                    'pytorch': torch.__version__,
                }

        with open(os.path.join(path, 'config_sentence_transformers.json'), 'w') as fOut:
            json.dump(self._model_config, fOut, indent=2)

        #Save modules
        for idx, name in enumerate(self._modules):
            module = self._modules[name]
            if idx == 0 and isinstance(module, Transformer):    #Save transformer model in the main folder
                model_path = path + "/"
            else:
                model_path = os.path.join(path, str(idx)+"_"+type(module).__name__)

            os.makedirs(model_path, exist_ok=True)
            module.save(model_path)
            modules_config.append({'idx': idx, 'name': name, 'path': os.path.basename(model_path), 'type': type(module).__module__})

        with open(os.path.join(path, 'modules.json'), 'w') as fOut:
            json.dump(modules_config, fOut, indent=2)

        # Create model card
        if create_model_card:
            self._create_model_card(path, model_name, train_datasets)

    def _create_model_card(self, path: str, model_name: Optional[str] = None, train_datasets: Optional[List[str]] = None):
        """
        Create an automatic model and stores it in path
        """
        if self._model_card_text is not None and len(self._model_card_text) > 0:
            model_card = self._model_card_text
        else:
            tags = ModelCardTemplate.__TAGS__.copy()
            model_card = ModelCardTemplate.__MODEL_CARD__

            # Print full model
            model_card = model_card.replace("{FULL_MODEL_STR}", str(self))

            # Add tags
            model_card = model_card.replace("{TAGS}", "\n".join(["- "+t for t in tags]))

            datasets_str = ""
            if train_datasets is not None:
                datasets_str = "datasets:\n"+"\n".join(["- " + d for d in train_datasets])
            model_card = model_card.replace("{DATASETS}", datasets_str)

            ###Add base model name
            model_card=model_card.replace("{BASE_MODEL}",self.base_model_name_or_path)

            ##ADd optional model description
            if self.opt_model_description is not None:
                model_card=model_card.replace("{MODEL_DESCRIPTION}",self.opt_model_description)
            else:
                model_card=model_card.replace("{MODEL_DESCRIPTION}","")
                
            ##ADd optional model language
            if self.opt_model_lang is not None:
                if isinstance(self.opt_model_lang,list):
                    model_card = model_card.replace("{LANGUAGE}", "\n".join(["- "+t for t in self.opt_model_lang]))
                else:
                    model_card = model_card.replace("{LANGUAGE}", "\n".join(["- "+t for t in [self.opt_model_lang]]))
            else:
                model_card = model_card.replace("language: \n{LANGUAGE}", "")
                model_card = model_card.replace("It is pretrained for the language : {LANGUAGE}.", "")

            # Add dim info
            self._model_card_vars["{NUM_DIMENSIONS}"] = self.get_sentence_embedding_dimension()

            # Replace vars we created while using the model
            for name, value in self._model_card_vars.items():
                model_card = model_card.replace(name, str(value))

            # Replace remaining vars with default values
            for name, value in ModelCardTemplate.__DEFAULT_VARS__.items():
                model_card = model_card.replace(name, str(value))

        if model_name is not None:
            model_card = model_card.replace("{MODEL_NAME}", model_name.strip())

        with open(os.path.join(path, "README.md"), "w", encoding='utf8') as fOut:
            fOut.write(model_card.strip())

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
            raise ValueError("You must login to the Hugging Face hub. Run `huggingface-cli login` or set `HF_TOKEN`.")

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
                except:
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
