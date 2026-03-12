from huggingface_hub import upload_folder
import os

# no stem
local_model_dir = "finetunes/scripts_models_dk-llm2vec-model2vec-dim256_sif0.0005_strip_upper_case_strip_exotic_focus_pca_normalize_embeddings-features_100000_max_length_800"
hf_token = os.getenv("HF_KEY")

upload_folder(
    repo_id="andersborges/model2vecdk",
    folder_path=local_model_dir,
    token=hf_token,
    create_pr=False,  # upload directly
)

# # stem
# local_model_dir = "finetunes/scripts_models_dk-llm2vec-model2vec-dim256_sif0.0001_strip_upper_case_strip_exotic_focus_pca_stem_normalize_embeddings-features_100000_max_length_800"
# hf_token = os.getenv("HF_KEY")

# upload_folder(
#     repo_id="andersborges/model2vecdk-stem",
#     folder_path=local_model_dir,
#     token=hf_token,
#     create_pr=False,  # upload directly
# )
