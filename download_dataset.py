import os
from huggingface_hub import snapshot_download

local_dir = "./dataset/isolated-guitar-chords"

print("="*60)
print(f"{'Downloading Dataset':^60}")
print("="*60)
print("Repository: rodriler/isolated-guitar-chords")
print("Size: ~1GB. This may take a while...\n")

os.makedirs(local_dir, exist_ok=True)

snapshot_download(
    repo_id="rodriler/isolated-guitar-chords",
    repo_type="dataset",
    local_dir=local_dir,
    resume_download=True
)

print("\nDownload completed successfully.")
print(f"The dataset has been stored at {local_dir}")