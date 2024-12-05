# set up the environment and install any missing packages:
#!pip install torch torchvision numpy scipy matplotlib pandas pillow tqdm MLclf

# PyTorch for building and training neural networks
import torch
from torch import nn, utils
import torch.nn.functional as F

# DataLoader for creating training and validation dataloaders
from torch.utils.data import DataLoader

# Torchvision for datasets and transformations
from torchvision import models, datasets, transforms

# Numpy for numerical operations
import numpy as np

# Matplotlib for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt

# Pandas for data manipulation
import pandas as pd

# PIL for image processing
from PIL import Image

# TQDM for progress bars
from tqdm import tqdm

# OS for operating system operations
import os

# Functions from utils to help with training and evaluation
from utils import (
    inspect_batch, test_evaluate_prototypes, training_plot, setup_dataset_prototype, 
    inspect_task, distillation_output_loss, evaluate_model_prototypes, get_batch_acc, logger
)

# Import the HyperCMTL_seq model architecture
from hypernetwork import HyperCMTL_seq, HyperCMTL_seq_simple, HyperCMTL_seq_prototype_simple

# Import the wandb library for logging metrics and visualizations
import wandb

### Learning without Forgetting:
from copy import deepcopy  # Deepcopy for copying models

# Time and logging for logging training progress
import time
import logging
import pdb

# Set random seeds for reproducibility
torch.manual_seed(0)
import random
random.seed(42)
np.random.seed(42)

# Clear CUDA cache
torch.cuda.empty_cache()

# Determine device
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


### Dataset hyperparameters:
VAL_FRAC = 0.1
TEST_FRAC = 0.1
BATCH_SIZE = 128
dataset = "Split-CIFAR100"  # Options: "Split-MNIST", "Split-CIFAR100", "TinyImageNet"
NUM_TASKS = 5 if dataset == 'Split-MNIST' else 10

### Training hyperparameters:
EPOCHS_PER_TIMESTEP = 15
lr     = 1e-4  # initial learning rate
l2_reg = 1e-6  # L2 weight decay term (0 means no regularisation)
temperature = 2.0  # temperature scaling factor for distillation loss
stability = 3  # `stability` term to balance this soft loss with the usual hard label loss for the current classification task.

# Create results directory
os.makedirs('results', exist_ok=True)
num = time.strftime("%m%d-%H%M%S")
results_dir = f'results/{num}-HyperCMTL_seq'
os.makedirs(results_dir, exist_ok=True)

# Initialize logger
logger = logger(results_dir)

# Log initial information
logger.log('Starting training...')
logger.log(f'Training hyperparameters: EPOCHS_PER_TIMESTEP={EPOCHS_PER_TIMESTEP}, lr={lr}, l2_reg={l2_reg}, temperature={temperature}, stability={stability}')
logger.log(f'Training on device: {device}')

### Define preprocessing transform and load dataset
data = setup_dataset_prototype(
    dataset_name=dataset, 
    data_dir='./data', 
    num_tasks=NUM_TASKS, 
    val_frac=VAL_FRAC, 
    test_frac=TEST_FRAC, 
    batch_size=BATCH_SIZE
)


# Extract data components
timestep_tasks = data['timestep_tasks']
final_test_loader = data['final_test_loader']
task_metadata = data['task_metadata']
task_test_sets = data['task_test_sets']
images_per_class = data['images_per_class']
timestep_task_classes = data['timestep_task_classes']
train_prototypes = data['train_prototype_image_per_class']
prototype_loader = data['prototype_loader']
task_prototypes = data['task_prototypes']
prototype_indices = data['prototype_indices']



# More complex model configuration
backbone = 'resnet50'  # Options: ['mobilenetv2', 'efficientnetb0', 'vit'] (vit not yet working)
task_head_projection_size = 512  # Even larger hidden layer in task head
hyper_hidden_features = 256  # Larger hypernetwork hidden layer size
hyper_hidden_layers = 4  # Deeper hypernetwork

# Initialize the model with the new configurations
model = HyperCMTL_seq_prototype_simple(
    num_instances=len(task_metadata),
    backbone=backbone,
    task_head_projection_size=task_head_projection_size,
    task_head_num_classes=len(task_metadata[0]),
    hyper_hidden_features=hyper_hidden_features,
    hyper_hidden_layers=hyper_hidden_layers,
    device=device,
    std=0.01
).to(device)

# Log the model architecture and configuration
logger.log(f'Model architecture: {model}')
logger.log(f"Model initialized with backbone_config={backbone}, task_head_projection_size={task_head_projection_size}, hyper_hidden_features={hyper_hidden_features}, hyper_hidden_layers={hyper_hidden_layers}")

# Initialize the previous model
previous_model = None

# Initialize optimizer and loss function:
opt = torch.optim.AdamW(model.get_optimizer_list(), lr=lr, weight_decay=l2_reg)
loss_fn = nn.CrossEntropyLoss()

### Metrics and plotting:
plot_training = True   # show training plots after each timestep
show_progress = True   # show progress bars and end-of-epoch metrics
verbose       = True   # output extra info to console

# Track metrics for plotting training curves:
metrics = { 
    'train_losses': [],
    'train_accs': [],
    'val_losses': [],
    'val_accs': [],
    'epoch_steps': [],  # used for plotting val loss at the correct x-position
    'CL_timesteps': [],  # used to draw where each new timestep begins
    'best_val_acc': 0.0,
    'steps_trained': 0,
    'soft_losses': [],  # distillation loss
}

prev_test_accs = []

print("Starting training")

with wandb.init(project='HyperCMTL', name=f'HyperCMTL_1seq-prototypes-{dataset}-{backbone}') as run:

    # Outer loop over each task, in sequence
    for t, (task_train, task_val) in tqdm(timestep_tasks.items(), desc="Processing Tasks"):
        logger.log(f"Training on task id: {t}  (classification between: {task_metadata[t]})")

        # Build train and validation loaders for the current task
        train_loader = DataLoader(
            task_train, 
            batch_size=BATCH_SIZE, 
            shuffle=True
        )
        val_loader = DataLoader(
            task_val, 
            batch_size=BATCH_SIZE, 
            shuffle=False
        )
        
        # Retrieve prototypes for the current task
        
        prototypes = task_prototypes[t].to(device) # Shape: (num_classes_per_task, C, H, W)
    

        # Inner loop over the current task:
        for e in range(EPOCHS_PER_TIMESTEP):
            epoch_train_losses, epoch_train_accs = [], []
            epoch_soft_losses = []

            # Initialize progress bar
            progress_bar = tqdm(train_loader, ncols=100, desc=f'Task {t} Epoch {e+1}') if show_progress else train_loader
            num_batches = len(train_loader)
            
            for batch_idx, batch in enumerate(progress_bar):
                # Get data from batch
                x, y, task_ids = batch
                x, y = x.to(device), y.to(device)
                task_id = task_ids[0]

                # Zero the gradients
                opt.zero_grad()

                # Forward pass: pass support set and prototypes to the model
                pred = model(x, prototypes, task_id).squeeze(0)

                # Compute hard loss
                hard_loss = loss_fn(pred, y)

                # Initialize soft loss
                soft_loss = torch.tensor(0.0).to(device)

                # Compute distillation loss if previous model exists
                if previous_model is not None:
                    for old_task_id in range(t):
                        with torch.no_grad():
                            # Previous model also needs to receive prototypes
                            old_pred = previous_model(x, prototypes, old_task_id)
                        # Current model's predictions for old tasks
                        new_prev_pred = model(x, prototypes, old_task_id)
                        # Accumulate distillation loss
                        soft_loss += distillation_output_loss(new_prev_pred, old_pred, temperature).mean().to(device)

                # Total loss
                total_loss = hard_loss + stability * soft_loss

                # Backward pass and optimization
                total_loss.backward()
                opt.step()

                # Calculate accuracy
                accuracy_batch = get_batch_acc(pred, y)

                # Log metrics to wandb
                wandb.log({
                    'hard_loss': hard_loss.item(), 
                    'soft_loss': (soft_loss * stability).item(), 
                    'train_loss': total_loss.item(), 
                    'epoch': e, 
                    'task_id': t, 
                    'batch_idx': batch_idx, 
                    'train_accuracy': accuracy_batch
                })

                # Track loss and accuracy
                epoch_train_losses.append(hard_loss.item())
                epoch_train_accs.append(accuracy_batch)
                epoch_soft_losses.append(soft_loss.item() if isinstance(soft_loss, torch.Tensor) else soft_loss)
                metrics['steps_trained'] += 1

                if show_progress:
                    # Update progress bar description
                    progress_bar.set_description(
                        f'E{e+1} batch loss:{hard_loss:.2f}, batch acc:{accuracy_batch:>5.1%}'
                    )

            # Evaluate after each epoch on the current task's validation set
            avg_val_loss, avg_val_acc = evaluate_model_prototypes(multitask_model=model,  # model to evaluate
                                    val_loader=val_loader, # task-specific data to evaluate on
                                    prototypes=prototypes ,  # prototypes for the current task
                                    task_id=task_id, # current task id
                                    loss_fn=loss_fn,  # loss function
                                    device=device)  # device to run on

            # Log validation metrics to wandb
            wandb.log({
                'val_loss': avg_val_loss, 
                'val_accuracy': avg_val_acc, 
                'epoch': e, 
                'task_id': t
            })

            # Update metrics
            metrics['epoch_steps'].append(metrics['steps_trained'])
            metrics['train_losses'].extend(epoch_train_losses)
            metrics['train_accs'].extend(epoch_train_accs)
            metrics['val_losses'].append(avg_val_loss)
            metrics['val_accs'].append(avg_val_acc)
            metrics['soft_losses'].extend(epoch_soft_losses)

            if show_progress:
                # Log end-of-epoch stats
                logger.log(
                    f'E{e+1} loss:{np.mean(epoch_train_losses):.2f}|v:{avg_val_loss:.2f}' +
                    f'| acc t:{np.mean(epoch_train_accs):>5.1%}|v:{avg_val_acc:>5.1%}'
                )

            # Update best validation accuracy
            if avg_val_acc > metrics['best_val_acc']:
                metrics['best_val_acc'] = avg_val_acc

        # Append current epoch steps
        metrics['CL_timesteps'].append(metrics['steps_trained'])

        # Plot training curves
        if plot_training and len(metrics['val_losses']) > 0:
            training_plot(metrics, show_timesteps=True, results_dir=f'{results_dir}/training-t{t}.png')

        if verbose:
            logger.log(f'Best validation accuracy: {metrics["best_val_acc"]:.2%}\n')
        metrics['best_val_acc'] = 0.0

        # Evaluate on all tasks up to current
        test_accs = test_evaluate_prototypes(
                        multitask_model=model, 
                        selected_test_sets=task_test_sets[:t+1],  
                        task_test_sets=task_test_sets, 
                        task_prototypes=task_prototypes, 
                        prev_accs=prev_test_accs,
                        show_taskwise_accuracy=True, 
                        baseline_taskwise_accs=None, 
                        model_name='HyperCMTL_seq + LwF', 
                        verbose=True, 
                        batch_size=BATCH_SIZE,
                        results_dir=results_dir,
                        task_metadata=task_metadata,
                        task_id=t,
                        loss_fn=loss_fn,
                    )

        # Log test accuracy to wandb
        wandb.log({'mean_test_acc': np.mean(test_accs), 'task_id': t})

        # Append test accuracies
        prev_test_accs.append(test_accs)
        '''
        # ----------- Prototype Integration Starts Here -----------

        # Retrieve prototypes for the current task
        # Shape: (num_classes_per_task, C, H, W)
        current_prototypes = task_prototypes[t].to(device)

        # Forward pass through the model to get prototype outputs
        proto_preds = model(current_prototypes, current_prototypes, t)  # Note: passing prototypes as support_set as well

        # Define prototype loss
        # Example: MSE Loss between current prototype predictions and stored prototype outputs
        # Initialize a dictionary to store prototype outputs per task
        if 'stored_proto_outputs' not in locals():
            stored_proto_outputs = {}

        if previous_model is not None and t > 0:
            # Retrieve stored prototype outputs from previous model
            stored_proto = stored_proto_outputs[t]
            # Compute prototype loss
            proto_loss = F.mse_loss(proto_preds, stored_proto)
            # Backward pass
            opt.zero_grad()
            proto_loss.backward()
            # Optimization step
            opt.step()
            # Log prototype loss
            logger.log(f'Task {t} Prototype loss: {proto_loss.item():.4f}')
            wandb.log({'prototype_loss': proto_loss.item(), 'task_id': t, 'epoch': e})

        # Store current prototype outputs for future tasks
        stored_proto_outputs[t] = proto_preds.detach()

        # ----------- Prototype Integration Ends Here -----------
        '''
        #store the current model as the previous model
        previous_model = model.deepcopy()
        previous_model.eval()  # Set to evaluation mode
        for param in previous_model.parameters():
            param.requires_grad = False  # Freeze the previous model

    # After all tasks, evaluate final test accuracy
    final_avg_test_acc = np.mean(test_accs)
    logger.log(f'Final average test accuracy: {final_avg_test_acc:.2%}')
    wandb.log({'final_avg_test_acc': final_avg_test_acc, 'task_id': t})
    wandb.summary['final_avg_test_acc'] = final_avg_test_acc