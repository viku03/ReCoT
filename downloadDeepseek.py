import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", save_dir=None):
    """
    Download TinyLlama model from Hugging Face with progress display using tqdm.
    
    Args:
        model_name (str): The name of the model on Hugging Face Hub.
        save_dir (str, optional): Directory to save the model. Defaults to model_name.
    """
    if save_dir is None:
        save_dir = model_name.split('/')[-1]
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Downloading model: {model_name}")
    print(f"This may take a while depending on your internet connection...")
    
    # First download the tokenizer with progress bar
    print("Downloading tokenizer...")
    with tqdm(desc="Tokenizer", unit="B", unit_scale=True) as pbar:
        AutoTokenizer.from_pretrained(
            model_name, 
            use_fast=True,
            cache_dir=save_dir,
            local_files_only=False
        )
    
    # Then download the model with progress bar
    print("Downloading model...")
    with tqdm(desc="Model", unit="B", unit_scale=True) as pbar:
        AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            cache_dir=save_dir,
            local_files_only=False
        )
    
    print(f"Model successfully downloaded and saved to: {save_dir}")
    
if __name__ == "__main__":
    download_model()