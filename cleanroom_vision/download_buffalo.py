from huggingface_hub import hf_hub_download
import os, zipfile

model_dir = os.path.expanduser("~/.insightface/models/buffalo_l")
os.makedirs(model_dir, exist_ok=True)

# Download the ZIP from the Hub
zip_path = hf_hub_download(
    repo_id="deepinsight/insightface-models",
    filename="buffalo_l.zip",
    local_dir=model_dir,
    local_dir_use_symlinks=False
)
print("Downloaded to", zip_path)

# Unzip in place
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(model_dir)
print("Extracted to", model_dir)

# Clean up
os.remove(zip_path)
print("Cleanup done.")
