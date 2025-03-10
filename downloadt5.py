import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def download_t5_base(cache_dir=None):
    """
    Downloads the T5-base model and tokenizer with progress tracking and error handling.
    Args:
        cache_dir (str, optional): Directory to store the downloaded model.
        If None, uses the default Hugging Face cache.
    Returns:
        tuple: (model, tokenizer) for T5-base
    """
    print("Starting T5-base download...")
    try:
        # Set up cache directory if provided
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            print(f"Created cache directory: {cache_dir}")
        
        # Download tokenizer with explicit local caching
        print("Downloading tokenizer...")
        tokenizer = T5Tokenizer.from_pretrained(
            "t5-base",
            cache_dir=cache_dir,
            local_files_only=False,
            use_fast=True
        )
        print("✓ Tokenizer downloaded successfully")
        
        # Download model with explicit local caching and progress tracking
        print("Downloading model (this may take a while)...")
        # Don't load the model into memory yet, just download the files
        T5ForConditionalGeneration.from_pretrained(
            "t5-base",
            cache_dir=cache_dir,
            local_files_only=False,
            torch_dtype=torch.float16,  # Use half precision
            return_dict=False,  # More memory efficient
        )
        print("✓ Model downloaded successfully")
        
        return tokenizer, cache_dir
    except Exception as e:
        print(f"Error during download: {str(e)}")
        # Provide troubleshooting suggestions
        print("\nTroubleshooting suggestions:")
        print("1. Check your internet connection")
        print("2. Try using a VPN if you're having connectivity issues with Hugging Face")
        print("3. Consider downloading via HTTP instead of HTTPS:")
        print("   export HF_HUB_DISABLE_IMPLICIT_TOKEN=1")
        print("   export HF_ENDPOINT=http://huggingface.co")
        print("4. Try alternative download methods like git-lfs:")
        print("   git lfs install")
        print("   git clone https://huggingface.co/t5-base")
        raise e

# Example usage
if __name__ == "__main__":
    # Set a custom cache directory in your project folder
    # or set to None to use default Hugging Face cache
    custom_cache_dir = "./models/t5_base_cache"
    tokenizer, model_path = download_t5_base(custom_cache_dir)
    
    print("\nModel files have been downloaded to:", model_path)
    print("T5-base is ready to use in your CoTGenerator class!")
    print("\nNote: To avoid memory issues, the model has not been loaded into memory.")
    print("It will be loaded when needed by your CoTGenerator class.")