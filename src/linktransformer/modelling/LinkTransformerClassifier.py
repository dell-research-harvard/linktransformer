from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Optional, Union, Iterable, List, Dict, Tuple, Callable, Any
import os
import logging
from huggingface_hub import HfApi, HfFolder, Repository
from sentence_transformers import __version__ 
import os
import json
import shutil
import stat
from linktransformer import __MODEL_HUB_ORGANIZATION__
import tempfile
from distutils.dir_util import copy_tree

from linktransformer.model_card_templates import ClassificationModelCardTemplate

logger = logging.getLogger(__name__)

##Define methods for 1) Saving model card with LT details, 2)Upload to huggingface (model + tokenizer)
class LinkTransformerClassifier:
    """Modified Sequence Classification model and tokenizer to implement model card generation and save to hub functions """
    def __init__(self, model_name_or_path: str,
                 opt_model_description: Optional[str] = None,
                 opt_model_lang: Optional[str] = None,
                 label_map=None,model_card_text=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.label_map = label_map
        self.model_name_or_path = model_name_or_path
        self.opt_model_description = opt_model_description
        self.opt_model_lang = opt_model_lang
        self._model_card_text = model_card_text
    

        ###Find the config file and get the base model name
        config_path = os.path.join(model_name_or_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as fIn:
                config = json.load(fIn)
                self.base_model_name_or_path = config['model_type']
        else:
            self.base_model_name_or_path = model_name_or_path



        ##If label map is none, try and find it in the model dir
        if self.label_map is None:
            label_map_path = os.path.join(model_name_or_path, 'label_map.json')
            if os.path.exists(label_map_path):
                with open(label_map_path, 'r') as fIn:
                    self.label_map = json.load(fIn)
        
    def save(self, save_directory: str,model_name: Optional[str] = None,override_model_description: Optional[str] = None, override_model_lang: Optional[str] = None,train_datasets=None):
        """
        Saves the model and tokenizer to the specified directory.
        """
        if override_model_description is not None:
            self.opt_model_description = override_model_description
        if override_model_lang is not None:
            self.opt_model_lang = override_model_lang
        self.tokenizer.save_pretrained(save_directory)
        self.model.save_pretrained(save_directory)
        self._create_model_card(save_directory, model_name=model_name,label_map=self.label_map,train_datasets=train_datasets)
        
    def _create_model_card(self, path: str, model_name: Optional[str] = None,label_map=None,train_datasets: Optional[List[str]] = None):
            """
            Create an automatic model and stores it in path
            """
            if self._model_card_text is not None and len(self._model_card_text) > 0:
                model_card = self._model_card_text
            else:
                tags = ClassificationModelCardTemplate.__TAGS__.copy()
                model_card = ClassificationModelCardTemplate.__MODEL_CARD__

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

                ##Add label map
                if label_map is not None:
                    label_map_str = ""
                    for key, value in label_map.items():
                        label_map_str += f"- {key}: {value}\n"
                    model_card = model_card.replace("{LABEL_MAP}", label_map_str)
                

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
                    model_card = model_card.replace("It is pretrained for the language: {LANGUAGE}.", "")



                # Replace remaining vars with default values
                for name, value in ClassificationModelCardTemplate.__DEFAULT_VARS__.items():
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
                    override_model_description: Optional[str] = None, 
                    override_model_lang: Optional[str] = None,
                    train_datasets: Optional[List[str]] = None):
        """
        Uploads all elements of this LinkTransformer (for classification) to a new HuggingFace Hub repository.

        :param repo_name: Repository name for your model in the Hub.
        :param organization:  Organization in which you want to push your model or tokenizer (you must be a member of this organization).
        :param private: Set to true, for hosting a prive model
        :param commit_message: Message to commit while pushing.
        :param local_model_path: Path of the model locally. If set, this file path will be uploaded. Otherwise, the current model will be uploaded
        :param exist_ok: If true, saving to an existing repository is OK. If false, saving only to a new repository is possible
        :param replace_model_card: If true, replace an existing model card in the hub with the automatically created model card
        :param train_datasets: Datasets used to train the model. If set, the datasets will be added to the model card in the Hub.
        :return: The url of the commit of your model in the given repository.
        """
        token = HfFolder.get_token()
        if token is None:
            raise ValueError("You must login to the Hugging Face hub on this computer by typing `transformers-cli login`.")

        if '/' in repo_name:
            splits = repo_name.split('/', maxsplit=1)
            if organization is None or organization == splits[0]:
                organization = splits[0]
                repo_name = splits[1]
            else:
                raise ValueError("You passed and invalid repository name: {}.".format(repo_name))

        endpoint = "https://huggingface.co"
        repo_id = repo_name
        if organization:
            repo_id = f"{organization}/{repo_id}"
        repo_url = HfApi(endpoint=endpoint).create_repo(
                repo_id=repo_id,
                token=token,
                private=private,
                repo_type=None,
                exist_ok=exist_ok,
            )
        full_model_name = repo_url[len(endpoint)+1:].strip("/")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # First create the repo (and clone its content if it's nonempty).
            logger.info("Create repository and clone it if it exists")
            repo = Repository(tmp_dir, clone_from=repo_url)

            # If user provides local files, copy them.
            if local_model_path:
                copy_tree(local_model_path, tmp_dir)
            else:  # Else, save model directly into local repo.
                self.save(tmp_dir, full_model_name,override_model_description, override_model_lang,train_datasets)

            ##Save the model config (_lt_model_config) to the repo if it exists
            if hasattr(self, '_lt_model_config'):
                with open(os.path.join(tmp_dir, 'LT_training_config.json'), 'w') as fOut:
                    json.dump(self._lt_model_config, fOut, indent=2)

            #Find files larger 5M and track with git-lfs
            large_files = []
            for root, dirs, files in os.walk(tmp_dir):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, tmp_dir)

                    if os.path.getsize(file_path) > (5 * 1024 * 1024):
                        large_files.append(rel_path)

            if len(large_files) > 0:
                logger.info("Track files with git lfs: {}".format(", ".join(large_files)))
                repo.lfs_track(large_files)

            logger.info("Push model to the hub. This might take a while")
            push_return = repo.push_to_hub(commit_message=commit_message)

            def on_rm_error(func, path, exc_info):
                # path contains the path of the file that couldn't be removed
                # let's just assume that it's read-only and unlink it.
                try:
                    os.chmod(path, stat.S_IWRITE)
                    os.unlink(path)
                except:
                    pass

            # Remove .git folder. On Windows, the .git folder might be read-only and cannot be deleted
            # Hence, try to set write permissions on error
            try:
                for f in os.listdir(tmp_dir):
                    shutil.rmtree(os.path.join(tmp_dir, f), onerror=on_rm_error)
            except Exception as e:
                logger.warning("Error when deleting temp folder: {}".format(str(e)))
                pass


        return push_return


