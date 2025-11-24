# download_model.py
import os
from huggingface_hub import snapshot_download

def download_base_model():
    """
    Downloads the base model weights from Hugging Face Hub.
    """
    # --- Configuration ---
    # The repository ID of the model on Hugging Face.
    # repo_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    repo_id = "meta-llama/Llama-3.1-8B-Instruct"
    
    # The local directory where the model will be saved.
    # It's recommended to use an absolute path and a location with ample storage.
    # The .gitignore is already configured to ignore '/mnt/data/'.
    out_dir = "/home/rs_students/nlpG4/deepseek_aligner/models/Llama-3.1-8B-Instruct"
    
    # A regular expression to filter which files to download.
    # This pattern downloads only the essential files:
    # - .safetensors: The memory-safe weight files.
    # - config.json: The model's configuration file.
    # - tokenizer.*: All files related to the tokenizer.
    # allow_patterns = ["*.safetensors", "*config.json", "tokenizer.*"]

    # --- Execution ---
    print(f"Starting download of model '{repo_id}'...")
    print(f"Saving to: {out_dir}")
    
    # Ensure the output directory exists.
    os.makedirs(out_dir, exist_ok=True)

    # Use huggingface_hub's snapshot_download to get the files.
    # local_dir_use_symlinks=False ensures files are copied directly,
    # which is more robust for many environments.
    snapshot_download(
        repo_id=repo_id,
        local_dir=out_dir,
        # allow_patterns=allow_patterns,
        local_dir_use_symlinks=False
    )
    
    print("-" * 50)
    print("✅ Model download complete.")
    print(f"Files are saved in: {out_dir}")
    print("-" * 50)

if __name__ == "__main__":
    # To run this script:
    # 1. Make sure you have huggingface_hub installed: pip install huggingface-hub
    # 2. If the model is private, log in: huggingface-cli login
    # 3. Execute the script: python download_model.py
    download_base_model()
