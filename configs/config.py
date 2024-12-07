### HyperCMTL training Configuration File

# 1. Dataset Parameters
# ----------------------
dataset_config = {
    "dataset": "Split-CIFAR100",  # Dataset used for training. You can switch to "Split-MNIST" or other datasets.
    "NUM_TASKS": 10,  # Number of tasks for the dataset. Typically 5 for Split-MNIST and 10 for Split-CIFAR100.
    "BATCH_SIZE": 128,  # Batch size used during training.
    "VAL_FRAC": 0.1,  # Fraction of the dataset to be used for validation.
    "TEST_FRAC": 0.1,  # Fraction of the dataset to be used for testing.
}

# 2. Model Hyperparameters
# ------------------------
model_config = {
    "backbone": "resnet50",  # Backbone architecture used for the model (e.g., "resnet50").
    "task_head_projection_size": 512,  # The size of the task-specific projection layer.
    "hyper_hidden_features": 256,  # The number of hidden features in the hypernetwork.
    "hyper_hidden_layers": 4,  # The number of hidden layers in the hypernetwork.
}

# 3. Training Parameters
# -----------------------
training_config = {
    "EPOCHS_PER_TIMESTEP": 12,  # Number of epochs per timestep (task).
    "lr": 1e-4,  # Learning rate for the optimizer (can be tuned via Optuna).
    "temperature": 2.0,  # Temperature for distillation loss (used in knowledge distillation).
    "stability": 1.0,  # Stability weight for soft distillation loss.
    "l2_reg": 0.0,  # L2 regularization coefficient (currently unused).
    "weight_hard_loss_prototypes": 0.2,  # Weight for the hard loss applied to the prototypes.
    "weight_soft_loss_prototypes": 0.05,  # Weight for the soft loss applied to the prototypes.
}

# 4. Optimizer & Loss Function
# ----------------------------
optimizer_config = {
    "optimizer": "AdamW",  # Optimizer used for training. AdamW is used here.
    "loss_fn": "CrossEntropyLoss",  # Loss function used. CrossEntropyLoss is used for classification.
}

# 5. Logging and Visualization Parameters
# ---------------------------------------
logging_config = {
    "log_file": "training.log",  # Log file where training information will be saved.
    "log_level": "INFO",  # Logging level for the training process (can be INFO, DEBUG, etc.).
    "plot_training": True,  # Whether to plot the training curves.
    "show_progress": True,  # Whether to show progress bars during training.
    "verbose": True,  # Whether to show detailed logs for each epoch.
}

# 6. Miscellaneous Parameters
# ---------------------------
misc_config = {
    "device": "cuda:4",  # Device for training (use "cpu" if no GPU is available).
    "results_dir": "results",  # Directory to store results.
    "wandb_project": "HyperCMTL",  # The name of the WandB project to track experiments.
}

# 7. Evaluation Parameters
# ------------------------
evaluation_config = {
    "eval_frequency": 1,  # Frequency of evaluation (e.g., every epoch).
    "plot_results": True,  # Whether to plot results after each timestep.
}

# 8. Optuna Hyperparameter Optimization Parameters
# -------------------------------------------------
optuna_config = {
    "direction": "maximize",  # Whether to maximize or minimize the objective (e.g., validation accuracy).
    "n_trials": 20,  # Number of trials for the optimization process.
}

# 9. Hyperparameter Config to Optuna
# ----------------------------------
optuna_params = {
    "lr": 1e-4,  # Initial learning rate, will be tuned by Optuna.
    "temperature": 2.0,  # Initial temperature for distillation, will be tuned by Optuna.
    "stability": 1.0,  # Stability parameter for soft loss, will be tuned by Optuna.
    "task_head_projection_size": 512,  # Projection size for task-specific heads.
    "hyper_hidden_features": 256,  # Number of features for the hidden layers in the hypernetwork.
    "hyper_hidden_layers": 4,  # Number of hidden layers in the hypernetwork.
}

# Combine all config sections into one dictionary for easy access
config = {
    "dataset": dataset_config,
    "model": model_config,
    "training": training_config,
    "optimizer": optimizer_config,
    "logging": logging_config,
    "misc": misc_config,
    "evaluation": evaluation_config,
    "optuna": optuna_config,
    "optuna_params": optuna_params
}
