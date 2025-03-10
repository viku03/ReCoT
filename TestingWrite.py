# Check if the directory exists and has write permissions
import os
import logging
import torch

# Config class to hold hyperparameters
class Config:
    def __init__(self):
        # Base Model
        self.model_name = "bert-base-uncased"  # can be replaced with any suitable LLM
        self.tokenizer_name = "bert-base-uncased"
        
        # Vision Model
        self.vision_model_name = "openai/clip-vit-base-patch32"
        
        # ScienceQA Dataset
        self.file_path = "/Users/Viku/Datasets/ScienceQA"
        self.train_path = "/Users/Viku/Datasets/ScienceQA/train/train.json"
        self.val_path = "/Users/Viku/Datasets/ScienceQA/val/val.json"
        self.max_seq_length = 512
        self.batch_size = 4
        
        # Training
        self.learning_rate = 5e-5
        self.weight_decay = 0.01
        self.epochs = 3
        self.warmup_steps = 100
        self.max_grad_norm = 1.0
        self.gradient_accumulation_steps = 8
        
        # RL Training
        self.ppo_epochs = 4
        self.reward_scale = 0.01
        self.clip_param = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        
        # Transformer Refiner
        self.refiner_model_name = "bert-base-uncased"  # Can be smaller than main model
        self.refiner_learning_rate = 2e-5
        self.refiner_weight_decay = 0.01
        self.refiner_batch_size = 16
        self.refiner_epochs = 2
        self.refiner_max_seq_length = 256  # Can be shorter than main model
        
        # Retrieval
        self.retrieval_top_k = 3
        self.embedding_dim = 768
        
        # Reflection
        self.reflection_threshold = 0.7
        
        # Paths
        self.output_dir = "outputs/"
        self.checkpoint_dir = "checkpoints/"
        self.exemplar_path = "exemplars.json"
        
        # Device
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")

        # Self Training
        self.max_answer_length = 64
        self.rl_updates = 1000
        self.self_training_iterations = 3

config = Config()

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Add file logging
if not os.path.exists(config.output_dir):
    os.makedirs(config.output_dir)

file_handler = logging.FileHandler(os.path.join(config.output_dir, 'debug.log'))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

print(f"Output directory: {config.output_dir}")
print(f"Directory exists: {os.path.exists(config.output_dir)}")



# Try to write a test file to confirm permissions
try:
    test_path = os.path.join(config.output_dir, "test_write.txt")
    with open(test_path, 'w') as f:
        f.write("Test write access")
    print(f"Successfully wrote to {test_path}")
    # Clean up
    os.remove(test_path)
except Exception as e:
    print(f"Failed to write: {e}")

# Check if any log messages are actually being generated
logger.info("TEST LOG MESSAGE")

# Force flush the log handlers to ensure writing to disk
for handler in logger.handlers:
    handler.flush()