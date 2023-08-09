import linktransformer as lt


model_path=""

###Load the model
model=lt.load_model(model_path)

model.save_to_hub(repo_name = "linktransformer-models-test", ##Write model name here
                    organization= "dell-research-harvard",
                    private = None,
                    commit_message = "Add new LinkTransformer model.",
                    local_model_path = None,
                    exist_ok = True,
                    replace_model_card = True,
                    )

