import linktransformer as lt


model_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/models/test_model/0"

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

