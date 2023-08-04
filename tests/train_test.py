import os
import linktransformer as lt
from linktransformer.data import DATA_DIR_PATH

def test_train_model():
    # Define the path to the test dataset
    dataset_path = os.path.join(DATA_DIR_PATH, "es_mexican_products.xlsx")

    # Call the train_model function
    saved_model_path = lt.train_model(
        model_path="hiiamsid/sentence_similarity_spanish_es",
        data=dataset_path,
        left_col_names=["description47"],
        right_col_names=['description48'],
        left_id_name=['tariffcode47'],
        right_id_name=['tariffcode48'],
        log_wandb=False,
        training_args={"num_epochs": 0,
                       "test_at_end": False,
                       "save_val_test_pickles": True,
                       "model_save_name": "test_model",
                       "opt_model_description": "test",
                       "opt_model_lang":"es"}
    )

    # Add assertions to check if the training was successful and the model was saved
    assert os.path.exists(saved_model_path), "Model not saved"
    # You can add more specific assertions based on your requirements


if __name__ == "__main__":
    test_train_model()