import os
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_mistral_model(
    model_name="TheBloke/Mistral-7B-v0.1-AWQ",
    local_dir="./models/mistral_cache",
    hf_token=None,
    load_in_8bit=True
):
    """
    Download and cache the Mistral model with proper authentication.
    
    Args:
        model_name: Hugging Face model name
        local_dir: Directory to save the model
        hf_token: Hugging Face API token (or will prompt if None)
        load_in_8bit: Whether to load in 8-bit precision
    
    Returns:
        tuple: (model, tokenizer)
    """
    # Create directory if needed
    os.makedirs(local_dir, exist_ok=True)
    
    # Check if token is provided, otherwise prompt for it
    if hf_token is None:
        print("You need to provide a Hugging Face token to access this model.")
        print("Get your token from: https://huggingface.co/settings/tokens")
        hf_token = input("Enter your Hugging Face token: ")
    
    # Login to Hugging Face
    try:
        login(token=hf_token)
        print("Successfully logged in to Hugging Face!")
    except Exception as e:
        print(f"Authentication failed: {e}")
        return None, None
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    try:
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=local_dir,
            use_fast=True,
            token=hf_token
        )
        
        # Make sure padding token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"Tokenizer downloaded and saved to {local_dir}")
        
        # Download model with quantization if appropriate
        model_kwargs = {
            "cache_dir": local_dir,
            "low_cpu_mem_usage": True,
            "token": hf_token
        }
        
        print(f"Downloading model {model_name}...")
        print("This may take a while depending on your internet connection and computer.")
        print("The model is around 14GB in size.")
        
        if load_in_8bit and (device.type == "cuda"):
            model_kwargs["load_in_8bit"] = True
            print("Loading model in 8-bit quantization")
        elif device.type != "cpu":
            model_kwargs["torch_dtype"] = torch.float16
            print("Loading model in float16")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        if not (load_in_8bit and device.type == "cuda"):  # Only move if not 8-bit
            model = model.to(device)
            
        print(f"Model downloaded and moved to {device}")
        print(f"Model files saved to {local_dir}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your Hugging Face token is valid")
        print("2. Check your internet connection")
        print("3. Ensure you have enough disk space (~15GB)")
        print("4. Make sure you have access to this model on Hugging Face")
        
        return None, None

if __name__ == "__main__":
    print("=" * 70)
    print("Mistral-7B Model Downloader")
    print("=" * 70)
    print("This script will help you download the Mistral-7B model with proper authentication.")
    print("You need a Hugging Face account with access to the model.")
    print("Get your token from: https://huggingface.co/settings/tokens")
    
    # Optional: Get token from command line
    import argparse
    parser = argparse.ArgumentParser(description="Download Mistral model with authentication")
    parser.add_argument("--token", type=str, help="Hugging Face token")
    parser.add_argument("--dir", type=str, default="./models/mistral_cache", help="Directory to save model")
    parser.add_argument("--no-8bit", action="store_true", help="Disable 8-bit quantization")
    args = parser.parse_args()
    
    # Download the model
    model, tokenizer = download_mistral_model(
        hf_token=args.token,
        local_dir=args.dir,
        load_in_8bit=not args.no_8bit
    )
    
    if model is not None:
        print("\nSuccess! Model and tokenizer have been downloaded.")
        print(f"You can now use them from directory: {args.dir}")
        print("\nExample usage:")
        print("```python")
        print("from transformers import AutoModelForCausalLM, AutoTokenizer")
        print(f"tokenizer = AutoTokenizer.from_pretrained('{args.dir}')")
        print(f"model = AutoModelForCausalLM.from_pretrained('{args.dir}')")
        print("```")
    else:
        print("\nFailed to download the model. Please check the error messages above.")