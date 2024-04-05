import linktransformer as lt
import os
from huggingface_hub import HfApi

def delete_hf_model_repo(repo_name):
    api = HfApi()
    api.delete_repo(repo_name)
    
def empty_hf_model_repo(repo_name):
    api = HfApi()
    api.empty_repo(repo_name)
    


def upload_local_model(path,hf_path,commit_message):
    model=lt.load_model(path)
    model.save_to_hub(repo_name = hf_path, ##Write model name here
                    organization= "dell-research-harvard",
                    private= None,
                    commit_message= commit_message,
                    local_model_path= path,
                    exist_ok= True,
                    replace_model_card= True,
                    replace_all_contents= True,
                    retain_model_card= True,
                    )
    
                    
##We want to ensure reproducability by ensuring that local models are same as online models.
##Make a dict with local path and HF path
##Then iterate through the dict and upload the models

wiki_models_dir="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/wiki_data/models"
un_models_dir="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/un_data/models"
japanese_dir="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/historicjapanese/models"
mexican_dir="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/data_outside_package/historicmexican/models"

model_dict = {
    ###Companies
    os.path.join(wiki_models_dir,"linkage_es_aliases"): "dell-research-harvard/lt-wikidata-comp-es",
    os.path.join(wiki_models_dir,"linkage_en_aliases"): "dell-research-harvard/lt-wikidata-comp-en",
    os.path.join(wiki_models_dir,"linkage_fr_aliases"): "dell-research-harvard/lt-wikidata-comp-fr",
    os.path.join(wiki_models_dir,"linkage_ja_aliases"): "dell-research-harvard/lt-wikidata-comp-ja",
    os.path.join(wiki_models_dir,"linkage_zh_aliases"): "dell-research-harvard/lt-wikidata-comp-zh",
    os.path.join(wiki_models_dir,"linkage_de_aliases"): "dell-research-harvard/lt-wikidata-comp-de",
    os.path.join(wiki_models_dir,"linkage_multi_aliases"): "dell-research-harvard/lt-wikidata-comp-multi",
    
    ##UN
    os.path.join(un_models_dir,"linkage_un_data_en_fine_coarse"): "dell-research-harvard/lt-un-data-fine-coarse-en",
    os.path.join(un_models_dir,"linkage_un_data_es_fine_coarse"): "dell-research-harvard/lt-un-data-fine-coarse-es",
    os.path.join(un_models_dir,"linkage_un_data_fr_fine_coarse"): "dell-research-harvard/lt-un-data-fine-coarse-fr",
        
    os.path.join(un_models_dir,"linkage_un_data_en_fine_fine"): "dell-research-harvard/lt-un-data-fine-fine-en",
    os.path.join(un_models_dir,"linkage_un_data_es_fine_fine"): "dell-research-harvard/lt-un-data-fine-fine-es",
    os.path.join(un_models_dir,"linkage_un_data_fr_fine_fine"): "dell-research-harvard/lt-un-data-fine-fine-fr",
        
    os.path.join(un_models_dir,"linkage_un_data_en_fine_industry"): "dell-research-harvard/lt-un-data-fine-industry-en",
    os.path.join(un_models_dir,"linkage_un_data_es_fine_industry"): "dell-research-harvard/lt-un-data-fine-industry-es",
    os.path.join(un_models_dir,"linkage_un_data_fr_fine_industry"): "dell-research-harvard/lt-un-data-fine-industry-fr",
        
    os.path.join(un_models_dir,"linkage_un_data_multi_fine_fine"): "dell-research-harvard/lt-un-data-fine-fine-multi",
    os.path.join(un_models_dir,"linkage_un_data_multi_fine_coarse"): "dell-research-harvard/lt-un-data-fine-coarse-multi",
    os.path.join(un_models_dir,"linkage_un_data_multi_fine_industry"): "dell-research-harvard/lt-un-data-fine-industry-multi",
    
    ##japanese companies - done
    os.path.join(japanese_dir,"lt-historicjapanesecompanies-comp-prod-ind_onlinecontrastive_full"): "dell-research-harvard/lt-historicjapan-onlinecontrastive",
    os.path.join(japanese_dir,"lt-historicjapanesecompanies-comp-prod-ind_supcon_full"): "dell-research-harvard/lt-historicjapan-supcon",
    os.path.join(japanese_dir,"lt-historicjapanesecompanies-comp-prod-ind_onlinecontrastive_full"): "dell-research-harvard/historicjapan",
    os.path.join(japanese_dir,"lt-wikidata-comp-prod-ind-ja"): "dell-research-harvard/lt-wikidata-comp-prod-ind-ja",
    
    ##mexican companies
    os.path.join(mexican_dir,"lt-mexicantrade4748"): "dell-research-harvard/lt-mexicantrade4748",


}


##Now, upload the models
for path in model_dict:
    print(f"Uploading {path} to {model_dict[path]}")
    upload_local_model(path,model_dict[path],
                       "Updated model with better training and evaluation. Test and val data included as pickle files. Older Legacy files were removed to avoid confusion.",
                       )
