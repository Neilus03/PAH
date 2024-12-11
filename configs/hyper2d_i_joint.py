### HyperCMTL training Configuration File
import argparse
import sys
import os
# Add the root of the project 
root = os.path.dirname(os.path.abspath(__file__))

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


# 1. Dataset Parameters
# ----------------------
dataset_config = {
    "dataset": args.dataset,  # Dataset used for training. You can switch to "Split-MNIST" or other datasets.
    "NUM_TASKS": args.num_tasks,  # Number of tasks for the dataset. Typically 5 for Split-MNIST and 10 for Split-CIFAR100.
    "BATCH_SIZE": args.batch_size,  # Batch size used during training.
    "VAL_FRAC": 0.1,  # Fraction of the dataset to be used for validation.
    "TEST_FRAC": 0.1,  # Fraction of the dataset to be used for testing.
    'data_dir': os.path.join(root, 'data') if not args.data_maria else "/ghome/mpilligua/AdvancedProject/TSR/data",
}

lr_config = {
                "hyper_emb": 1e-3,  # Learning rate for the hyper-embedding network.
                "hyper_emb_reg": 1e-4,  # Learning rate for the hyper-embedding network.
                "backbone": 1e-4,  # Learning rate for the backbone network.
                "backbone_reg": 1e-5,  # Learning rate for the backbone network.
                "task_head": 1e-4,  # Learning rate for the task head.
                "task_head_reg": 1e-3,  # Learning rate for the task head.
                "hypernet": 1e-4,  # Learning rate for the hypernetwork.
                "hypernet_reg": 1e-3,  # Learning rate for the hypernetwork.
}

# 2. Model Hyperparameters
# ------------------------
model_config = {
    "backbone": args.backbone, # Backbone architecture used for the model (e.g., "resnet50").
    "hyper_hidden_features": 1024,
    "hyper_hidden_layers": 6,
    "frozen_backbone": args.frozen_backbone,  # Whether to freeze the backbone during training.
    "prototypes_channels": 1, # Number of channels of prototypes 1 for grayscale, 3 for RGB
    "prototypes_size": 20,  # Size of the prototypes.
    "initialize_prot_w_images": True,
    "mean_initialization_prototypes": 0.5,  # Mean for the initialization of the prototypes.
    "std_initialization_prototypes": 0.1,  # Standard deviation for the initialization of the prototypes.
    "lr_config": lr_config}


# 3. Training Parameters
# -----------------------
training_config = {
    "epochs_per_timestep": 12,  # Number of epochs per timestep (task).
    "temperature": args.temperature,  # Temperature for distillation loss (used in knowledge distillation).
    "stability":  args.stability,  # Stability weight for soft distillation loss.
    "weight_hard_loss_prototypes": 0.2,  # Weight for the hard loss applied to the prototypes.
    "weight_soft_loss_prototypes": 0.05,  # Weight for the soft loss applied to the prototypes.
    "weight_smoothness_loss": 0,
    "optimizer": "AdamW",  # Optimizer used for training. AdamW is used here.
}


# 5. Logging and Visualization Parameters
# ---------------------------------------
frozen = "frozen" if model_config["frozen_backbone"] else ""
name = f"Hyper2d_i_joint-{frozen}-{model_config['backbone']}-{dataset_config['dataset']}"
logging_config = {
    "log_file": "training.log",  # Log file where training information will be saved.
    "log_level": "INFO",  # Logging level for the training process (can be INFO, DEBUG, etc.).
    "plot_training": True,  # Whether to plot the training curves.
    "show_progress": True,  # Whether to show progress bars during training.
    "verbose": True,  # Whether to show detailed logs for each epoch.
    "results_dir": "results",  # Folder to save the results.
    "name": name,  # or EWC_Baseline pr SI_Baseline,
    "group": "Joint Incremental"
}

# 6. Miscellaneous Parameters
# ---------------------------
misc_config = {
    "device": args.device,  # Device for training (use "cpu" if no GPU is available).
    "seed": 42,  # Seed for reproducibility.
}

# 7. Evaluation Parameters
# ------------------------
evaluation_config = {
    "eval_frequency": 1,  # Frequency of evaluation (e.g., every epoch).
    "plot_results": True,  # Whether to plot results after each timestep.
}



# Combine all config sections into one dictionary for easy access
config = {
    "dataset": dataset_config,
    "model": model_config,
    "training": training_config,
    "logging": logging_config,
    "misc": misc_config,
    "evaluation": evaluation_config,
}
