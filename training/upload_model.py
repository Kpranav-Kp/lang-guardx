from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="models/distilbert",
    repo_id="KPranavKp/langguardx-distilbert",
    repo_type="model"
)
print("Uploaded successfully")