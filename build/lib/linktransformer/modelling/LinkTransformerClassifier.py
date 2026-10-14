from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Optional, Union, Iterable, List, Dict, Tuple, Callable, Any
import os
import logging
from huggingface_hub import HfApi, get_token
from sentence_transformers import __version__ 
import os
import json
import shutil
import stat
from linktransformer.constants import __MODEL_HUB_ORGANIZATION__
import tempfile

from linktransformer.model_card_templates import render_classifier_card

logger = logging.getLogger(__name__)


def _copy_dir_contents(src_dir: str, dst_dir: str):
    for name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, name)
        dst_path = os.path.join(dst_dir, name)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)

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
    

        ###Find the base model name: train_clf_model records it in LT_training_config.json; configs saved by
        ###older transformers versions record it as _name_or_path
        self.base_model_name_or_path = None
        lt_config_path = os.path.join(model_name_or_path, 'LT_training_config.json')
        config_path = os.path.join(model_name_or_path, 'config.json')
        if os.path.exists(lt_config_path):
            with open(lt_config_path, 'r') as fIn:
                self.base_model_name_or_path = json.load(fIn).get('base_model_path')
        elif os.path.exists(config_path):
            with open(config_path, 'r') as fIn:
                self.base_model_name_or_path = json.load(fIn).get('_name_or_path') or None
        elif not os.path.isdir(model_name_or_path):
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

        :param save_directory: Directory to save to.
        :param model_name: Name of the model to save.
        :param override_model_description: Override the model description in the model card.
        :param override_model_lang: Override the model language in the model card.
        :param train_datasets: Datasets used to train the model. If set, the datasets will be added to the model card in the Hub.

        :return: None
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
        Write the model card (README.md) to path. A card passed in as model_card_text is used as is.
        """
        if self._model_card_text:
            model_card = self._model_card_text
        else:
            model_card = render_classifier_card(
                model_name=(model_name or path).strip(),
                base_model=self.base_model_name_or_path,
                languages=self.opt_model_lang,
                description=self.opt_model_description,
                label_map=label_map,
                train_datasets=train_datasets,
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
            # If user provides local files, copy them.
            if local_model_path:
                _copy_dir_contents(local_model_path, tmp_dir)
            else:  # Else, save model directly into local repo.
                self.save(tmp_dir, full_model_name,override_model_description, override_model_lang,train_datasets)

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


