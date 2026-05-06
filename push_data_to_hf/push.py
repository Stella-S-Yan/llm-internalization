import os
from huggingface_hub import HfApi
from huggingface_hub import login
import fnmatch

login(token="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")  #"hf_xxxxxx"
 
from huggingface_hub import whoami
print(whoami())

api = HfApi()

base = "/usr/local/google/home/stellasyan/Documents/workspace"

# # Upload model
# model_base = base + "/model"
# for root, dirs, files in os.walk(model_base):
#     for f in files:
#         if f.endswith(".tgz") and not fnmatch.fnmatch(f, "*merged_think_sft_model_*.tgz"):
#             local_path = os.path.join(root, f)
#             rel_path = os.path.relpath(local_path, base)

#             print(f)
#             print("here")
#             api.upload_file(
#                 path_or_fileobj=local_path,
#                 path_in_repo=f"model/{f}",
#                 repo_id="UsernameAlreadyExitsts/llm_internalization",
#                 repo_type="dataset",
#                 commit_message=f"Update {rel_path}",
#             )


# Upload processed_data
data_base = base + "/processed_data"
for root, dirs, files in os.walk(data_base):
    for f in files:
        if f.startswith("Lepard_10k"):  # "Amazon", "Lepard_10k"
            local_path = os.path.join(root, f)
            rel_path = os.path.relpath(local_path, base)

            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=f"processed_data/{f}",
                repo_id="UsernameAlreadyExitsts/llm_internalization",
                repo_type="dataset",
                commit_message=f"Update {rel_path}",
            )

# Upload a final model
model_base = base + "/model"
for root, dirs, files in os.walk(model_base):
    for f in files:
        if fnmatch.fnmatch(f, "*merged_think_*.tgz"):
            local_path = os.path.join(root, f)
            rel_path = os.path.relpath(local_path, base)

            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=f"final_model/{f}",
                repo_id="UsernameAlreadyExitsts/llm_internalization",
                repo_type="dataset",
                commit_message=f"Update {rel_path}",
            )