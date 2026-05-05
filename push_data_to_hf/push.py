import os
from huggingface_hub import HfApi
from huggingface_hub import login

login(token="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")  #"hf_xxxxxx"
 
from huggingface_hub import whoami
print(whoami())

api = HfApi()

base = "/usr/local/google/home/stellasyan/Documents/workspace/processed_data"


# Upload model
model_base = base + "/model"
for root, dirs, files in os.walk(model_base):
    for f in files:
        if f.endswith(".tgz"):
            local_path = os.path.join(root, f)
            rel_path = os.path.relpath(local_path, base)

            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=f"model/{rel_path}",
                repo_id="UsernameAlreadyExitsts/llm_internalization",
                repo_type="dataset",
                commit_message=f"Update {rel_path}",
            )


# # Upload processed_data
# data_base = base + "/processed_data"
# for root, dirs, files in os.walk(data_base):
#     for f in files:
#         if f.startswith("Amazon"):
#             local_path = os.path.join(root, f)
#             rel_path = os.path.relpath(local_path, base)

#             api.upload_file(
#                 path_or_fileobj=local_path,
#                 path_in_repo=f"processed_data/{rel_path}",
#                 repo_id="UsernameAlreadyExitsts/llm_internalization",
#                 repo_type="dataset",
#                 commit_message=f"Update {rel_path}",
#             )

# # Upload a single file
# api.upload_file(
#     path_or_fileobj="/usr/local/google/home/stellasyan/Documents/workspace/model/Amazon_Beauty_think_sft_adaptor_0.tgz",
#     path_in_repo="final_model/Amazon_Beauty_think_sft_adaptor_0.tgz",
#     repo_id="UsernameAlreadyExitsts/llm_internalization",
#     repo_type="dataset",
#     commit_message="Uploading amazon models"
# )