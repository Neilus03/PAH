### HyperCMTL training Configuration File

import sys
import os
# Add the root of the project 
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 1. Dataset Parameters
# ----------------------
dataset_config = {
    "dataset": "Split-CIFAR100",  # Dataset used for training. You can switch to "Split-MNIST" or other datasets.
    "NUM_TASKS": 10,  # Number of tasks for the dataset. Typically 5 for Split-MNIST and 10 for Split-CIFAR100.
    "BATCH_SIZE": 256,  # Batch size used during training.
    "VAL_FRAC": 0.1,  # Fraction of the dataset to be used for validation.
    "TEST_FRAC": 0.1,  # Fraction of the dataset to be used for testing.
    'data_dir': os.path.join(root, 'data')
}

# 2. Model Hyperparameters
# ------------------------
model_config = {
    "backbone": "resnet50",  # Backbone architecture used for the model (e.g., "resnet50").
    "task_head_projection_size": 512,  # The size of the task-specific projection layer.}
    "frozen_backbone": False,  # Whether to freeze the backbone during training.
}

# 3. Training Parameters
# -----------------------
training_config = {
    "epochs_per_timestep": 12,  # Number of epochs per timestep (task).
    "lr": 1e-4,  # Learning rate for the optimizer (can be tuned via Optuna).
    "l2_reg": 1e-6,  # L2 regularization coefficient (currently unused).
    "temperature": 2.0,  # Temperature for distillation loss (used in knowledge distillation).
    "stability": 3,  # Stability weight for soft distillation loss.
    "ewc_lambda": 5000,  # Lambda hyperparameter for EWC.
    "si_epsilon": 1e-5,  # Epsilon hyperparameter for SI.
    "si_lambda": 1.0,  # Lambda hyperparameter for SI.
    "weight_hard_loss_prototypes": 0.2,  # Weight for the hard loss applied to the prototypes.
    "weight_soft_loss_prototypes": 0.05,  # Weight for the soft loss applied to the prototypes.
    "freeze_backbone": False,  # Whether to freeze the backbone during training.
    "backbone": "resnet50",  # to choose from resnet50, mobilenetv2, efficientnetb0
    "optimizer": "AdamW",  # Optimizer used for training. AdamW is used here.
}

# 5. Logging and Visualization Parameters
# ---------------------------------------
frozen = "frozen" if model_config["frozen_backbone"] else ""
name = f"SI baseline-{frozen}-{model_config['backbone']}-{dataset_config['dataset']}"
logging_config = {
    "log_file": "training.log",  # Log file where training information will be saved.
    "log_level": "INFO",  # Logging level for the training process (can be INFO, DEBUG, etc.).
    "plot_training": True,  # Whether to plot the training curves.
    "show_progress": True,  # Whether to show progress bars during training.
    "verbose": True,  # Whether to show detailed logs for each epoch.
    "results_dir": "results",  # Folder to save the results.
    "name": name,  # or SI_Baseline pr SI_Baseline
}

# 6. Miscellaneous Parameters
# ---------------------------
misc_config = {
    "device": "cuda:2",  # Device for training (use "cpu" if no GPU is available).
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
