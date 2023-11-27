
import linktransformer as lt
import pandas as pd
import os


##Roberta large best config
training_config_dict_rl = {
    # "advice": {"lr": 1e-6, "batch_size": 8, "model": "roberta-large", "weighted_loss":False},
    # "antitrust": {"lr": 1e-6, "batch_size": 8, "model": "roberta-large", "weighted_loss":False},
    # "bible": {"lr": 1e-6, "batch_size": 8, "model": "roberta-large", "weighted_loss":False},
    # "civil_rights": {"lr": 1e-5, "batch_size": 8, "model": "roberta-large", "weighted_loss":False},
    # "contraception": {"lr": 1e-6, "batch_size": 8, "model": "roberta-large", "weighted_loss":False},
    # "crime": {"lr": 1e-6, "batch_size": 8, "model": "roberta-large", "weighted_loss":False},
    # "horoscope_classification": {"lr": 1e-6, "batch_size": 8, "model": "roberta-large", "weighted_loss":False},
    # "labor_movement": {"lr": 1e-6, "batch_size": 8, "model": "roberta-large", "weighted_loss":False},
    "obits_classification": {"lr": 1e-6, "batch_size": 8, "model": "roberta-large", "weighted_loss":False},
    # "pesticide": {"lr": 1e-6, "batch_size": 8, "model": "roberta-large", "weighted_loss":False},
    # "poliovac": {"lr": 1e-6, "batch_size": 8, "model": "roberta-large", "weighted_loss":False},
    # "politics": {"lr": 1e-6, "batch_size": 32, "model": "distilroberta-base", "weighted_loss":False},
    # "protests": {"lr": 1e-5, "batch_size": 8, "model": "distilroberta-base", "weighted_loss":False},
    # "red_scare": {"lr": 1e-6, "batch_size": 8, "model": "roberta-large", "weighted_loss":False},
    # "schedules": {"lr": 1e-6, "batch_size": 8, "model": "roberta-large", "weighted_loss":False},
    # "sport": {"lr": 1e-6, "batch_size": 8, "model": "roberta-large", "weighted_loss":False},
    # "vietnam_war": {"lr": 1e-6, "batch_size": 8, "model": "roberta-large", "weighted_loss":False},
    # "weather_classification": {"lr": 1e-5, "batch_size": 8, "model": "roberta-large", "weighted_loss":False},
    # "ww1": {"lr": 1e-6, "batch_size": 8, "model": "roberta-large", "weighted_loss":False},
}

folder_list=list(training_config_dict_rl.keys())



results={}
for file_name in folder_list:
    # if not os.path.exists(label_col_name):
    if file_name=="wandb":
        continue
    if file_name=="tmp_trainer":
        continue
    print(file_name)
# df=pd.read_csv('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/extra_data/violence_multimodal/2023_09_08_final_protest_image_labels_inner_ids_1_to_400_proc.csv')
    train_data=pd.read_csv(f'/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/extra_data/gpt_all_topics/{file_name}/train.csv')
    val_data=pd.read_csv(f'/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/extra_data/gpt_all_topics/{file_name}/eval.csv')
    test_data=pd.read_csv(f'/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/extra_data/all_topics/{file_name}/fine_tuning_data/test.csv')

    config_for_model=training_config_dict_rl[file_name]


    ##Batch size 16 for roberta large, 32 for distil/base ; 2e7, 200 epochs; 8 for some large - 1e-6 ones
    best_model_path,best_metric, label_map= lt.train_clf_model(train_data=train_data,val_data=val_data,test_data=test_data,model=config_for_model["model"],
                                                               on=["article"],label_col_name="label",
                                                               data_dir=f"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/gpt_labels_trained_incomplete/{file_name}/fine_tuning_data/",
                        lr=config_for_model["lr"],batch_size=config_for_model["batch_size"],
                        training_args={"num_train_epochs":50},
                        model_save_dir=f"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/gpt_labels_trained_incomplete/{file_name}/models/gpt_best_hp_redone/", weighted_loss=False,weight_list=None,
                        wandb_log=True,wandb_name=file_name)
    results[file_name]={"best_model_path":best_model_path,"best_metric":best_metric,"label_map":label_map}

    # pd.DataFrame(results).to_csv(f"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/gpt_labels_trained_incomplete/{file_name}/best_results_gpt_best_hp.csv")
