###Test classification
import linktransformer as lt
import os
import pandas as pd
import numpy as np




def test_class_train_splits():
    train_data = pd.read_csv(os.path.join(lt.DATA_DIR_PATH, 'train_weather.csv'))
    test_data = pd.read_csv(os.path.join(lt.DATA_DIR_PATH, 'test_weather.csv'))
    eval_data = pd.read_csv(os.path.join(lt.DATA_DIR_PATH, 'eval_weather.csv'))

    model="distilroberta-base"

    lt.train_clf_model(model=model,on=["article"],label_col_name="label",train_data=train_data,val_data=eval_data,test_data=test_data,data_dir=".",
                    training_args={},
                    eval_steps=None,batch_size=None,lr=2e-6,
                    epochs=10,model_save_dir="test_lt_clf", weighted_loss=False,weight_list=None,
                    wandb_log=False)



def test_class_train_nosplits():
    train_data = pd.read_csv(os.path.join(lt.DATA_DIR_PATH, 'train_weather.csv'))
    test_data = pd.read_csv(os.path.join(lt.DATA_DIR_PATH, 'test_weather.csv'))
    eval_data = pd.read_csv(os.path.join(lt.DATA_DIR_PATH, 'eval_weather.csv'))

    # model="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/test_lt_clf/checkpoint-330"
    model="distilroberta-base"

    lt.train_clf_model(data=train_data,model=model,on=["article"],label_col_name="label",data_dir=".",
                    training_args={},
                    eval_steps=None,batch_size=None,lr=2e-6,
                    epochs=2,model_save_dir="test_lt_clf", weighted_loss=False,weight_list=None,
                    wandb_log=False)

def test_class_train_3_labels():
    train_data = pd.read_csv(os.path.join(lt.DATA_DIR_PATH, 'protest_train.csv'))
    # test_data = pd.read_csv(os.path.join(lt.DATA_DIR_PATH, 'protest_test.csv'))
    # eval_data = pd.read_csv(os.path.join(lt.DATA_DIR_PATH, 'protest_val.csv'))

    # model="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/test_lt_clf/checkpoint-330"
    model="distilroberta-base"

    lt.train_clf_model(data=train_data,model=model,on=["headline","byline","text"],label_col_name="label",data_dir=".",
                    training_args={},
                    eval_steps=None,batch_size=None,lr=2e-5,
                    epochs=2,model_save_dir="test_lt_clf", weighted_loss=False,weight_list=None,
                    wandb_log=False)


def test_class_train_3_labels_weighted():
    train_data = pd.read_csv(os.path.join(lt.DATA_DIR_PATH, 'protest_train.csv'))
    # test_data = pd.read_csv(os.path.join(lt.DATA_DIR_PATH, 'protest_test.csv'))
    # eval_data = pd.read_csv(os.path.join(lt.DATA_DIR_PATH, 'protest_val.csv'))

    # model="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/test_lt_clf/checkpoint-330"
    model="distilroberta-base"

    lt.train_clf_model(data=train_data,model=model,on=["headline","byline","text"],label_col_name="label",data_dir=".",
                    training_args={},
                    eval_steps=None,batch_size=None,lr=2e-5,
                    epochs=2,model_save_dir="test_lt_clf", weighted_loss=True,weight_list=None,
                    wandb_log=False,print_test_mistakes=True)
##Run as script
if __name__ == "__main__":
    test_class_train_splits()
    # test_class_train_nosplits()
    # test_class_train_3_labels()
    # test_class_train_3_labels_weighted()