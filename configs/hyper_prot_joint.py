import argparse
import sys
import os

# Add the root of the project 
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define argument parser
parser = argparse.ArgumentParser(description="HyperCMTL Training Configuration")
parser.add_argument("-dataset", type=str, default="Split-CIFAR100", help="Dataset to use (e.g., Split-MNIST, Split-CIFAR100)")
parser.add_argument("-backbone", type=str, default="resnet50", help="Backbone architecture (e.g., resnet50, mobilenetv2)")
parser.add_argument("-frozen_backbone", action="store_true", help="Freeze the backbone during training")
parser.add_argument("-temperature", type=float, default=2.0, help="Temperature for distillation loss")
parser.add_argument("-stability", type=float, default=3, help="Stability weight for soft loss")
parser.add_argument("-batch_size", type=int, default=256, help="Batch size for training")
parser.add_argument("-device", type=str, default="cuda:0", help="Device to use for training (e.g., cuda:0, cpu)")
parser.add_argument("-data_maria", action="store_true", help="Set this flag if you are running the code on Maria")
parser.add_argument("-num_tasks", type=int, default=None, help="Number of tasks for the dataset. Typically 5 for Split-MNIST and 10 for Split-CIFAR100")

# Parse arguments
args = parser.parse_args()

if args.num_tasks is None:
    args.num_tasks = 10 if args.dataset == "Split-CIFAR100" else 5

# Dataset Configuration
dataset_config = {
    "dataset": args.dataset,
    "NUM_TASKS": args.num_tasks,
    "BATCH_SIZE": args.batch_size,
    "VAL_FRAC": 0.1,
    "TEST_FRAC": 0.1,
    'data_dir': os.path.join(root, 'data') if not args.data_maria else "/ghome/mpilligua/AdvancedProject/TSR/data",
}

# Learning Rate Configuration
lr_config = {
    "hyper_emb": 1e-3,
    "hyper_emb_reg": 1e-4,
    "backbone": 1e-4,
    "backbone_reg": 1e-5,
    "task_head": 1e-4,
    "task_head_reg": 1e-3,
    "hypernet": 1e-4,
    "hypernet_reg": 1e-3,
    "linear_prototypes": 1e-3,
    "linear_prototypes_reg": 1e-5,
}

# Model Configuration
model_config = {
    "backbone": args.backbone,
    "hyper_hidden_features": 1024,
    "hyper_hidden_layers": 6,
    "projection_prototypes": 4096,
    "frozen_backbone": args.frozen_backbone,
    "emb_size": 1024,
    "mean_initialization_emb": 0,
    "std_initialization_emb": 0.01,
    "lr_config": lr_config,
}

# Training Configuration
training_config = {
    "epochs_per_timestep": 12,
    "temperature": args.temperature,
    "stability": args.stability,
    "weight_hard_loss_prototypes": 0.2,
    "weight_soft_loss_prototypes": 0.05,
    "weight_smoothness_loss": 0,
    "optimizer": "AdamW",
}

# Logging Configuration
frozen = "frozen" if model_config["frozen_backbone"] else ""
name = f"Hyper-prot-joint{frozen}-{model_config['backbone']}-{dataset_config['dataset']}"
logging_config = {
    "log_file": "training.log",
    "log_level": "INFO",
    "plot_training": True,
    "show_progress": True,
    "verbose": True,
    "results_dir": "results",
    "name": name,
    "group": "Joint Incremental",
}

# Miscellaneous Configuration
misc_config = {
    "device": args.device,
    "seed": 42,
}

# Evaluation Configuration
evaluation_config = {
    "eval_frequency": 1,
    "plot_results": True,
}

# Combine all config sections into one dictionary
config = {
    "dataset": dataset_config,
    "model": model_config,
    "training": training_config,
    "logging": logging_config,
    "misc": misc_config,
    "evaluation": evaluation_config,
}

# Example: Print config to verify
if __name__ == "__main__":
    print(config)
