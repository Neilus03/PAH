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
from utils import *

# Import the HyperCMTL_seq model architecture
from networks.hypernetwork import HyperCMTL_seq, HyperCMTL_seq_simple, HyperCMTL_seq_prototype_simple

# Import the wandb library for logging metrics and visualizations
import wandb

### Learning without Forgetting:
from copy import deepcopy  # Deepcopy for copying models

# Time and logging for logging training progress
import time
import logging
import pdb
import sys

import random


config = config_load(sys.argv[1])["config"]
device = torch.device(config["misc"]["device"] if torch.cuda.is_available() else "cpu")
seed_everything(config['misc']['seed'])

num = time.strftime("%Y%m%d-%H%M%S")
name_run = f"{config['logging']['name']}-{num}"
results_dir = os.path.join(config["logging"]["results_dir"], name_run)
os.makedirs(results_dir, exist_ok=True)

# Initialize logger
logger = logger(results_dir)

# Log initial information
logger.log(f"Starting training for {config['logging']['name']}")
logger.log(f"Configuration: {config}")
logger.log(f"Device: {device}")
logger.log(f"Random seed: {config['misc']['seed']}")

### Define preprocessing transform and load dataset
if config["dataset"]["dataset"] != "TinyImageNet":
    data = setup_dataset(dataset_name = config["dataset"]["dataset"],
                        data_dir = config["dataset"]["data_dir"], 
                        num_tasks=config["dataset"]["NUM_TASKS"],
                        val_frac=config["dataset"]["VAL_FRAC"],
                        test_frac=config["dataset"]["TEST_FRAC"],
                        batch_size=config["dataset"]["BATCH_SIZE"])
else:
    data = setup_tinyimagenet(data_dir = config["dataset"]["data_dir"], 
                        num_tasks=config["dataset"]["NUM_TASKS"],
                        val_frac=config["dataset"]["VAL_FRAC"],
                        test_frac=config["dataset"]["TEST_FRAC"],
                        batch_size=config["dataset"]["BATCH_SIZE"])

# More complex model configuration

num_tasks = len(data['task_metadata'])
num_classes_per_task = len(data['task_metadata'][0])

logger.log(f"Using backbone: {config['model']['backbone']}")

# Initialize the model with the new configurations
model = HyperCMTL_seq_prototype_simple(
    num_tasks=num_tasks,
    num_classes_per_task=num_classes_per_task,
    model_config=config['model'],
    device=device
).to(device)

logger.log(f"Model created!")
logger.log(f"Model initialized with freeze_backbone={config['model']['frozen_backbone']}, config={config['model']}")

# Initialize the previous model
previous_model = None

# Initialize optimizer and loss function:
opt = torch.optim.AdamW(model.get_optimizer_list())
loss_fn = nn.CrossEntropyLoss()

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
logger.log(f"Starting training for {config['logging']['name']}")

with wandb.init(project='HyperCMTL', entity='pilligua2', name=f'{name_run}', config=config) as run:
    # Outer loop over each task, in sequence
    for t, (task_train, task_val) in data['timestep_tasks'].items():
        task_train.num_classes = len(data['timestep_task_classes'][t])
        logger.log(f"Task {t}: {task_train.num_classes} classes\n: {data['task_metadata'][t]}")
        
        # Build train and validation loaders for the current task
        train_loader = DataLoader(
            task_train, 
            batch_size=config['dataset']['BATCH_SIZE'], 
            shuffle=True
        )
        val_loader = DataLoader(
            task_val, 
            batch_size=config['dataset']['BATCH_SIZE'], 
            shuffle=False
        )
        
        # Retrieve prototypes for the current task
        prototypes = data["task_prototypes"][t].to(device) # Shape: (num_classes_per_task, C, H, W)
    
        # Inner loop over the current task:
        for e in range(config['training']['epochs_per_timestep']):
            epoch_train_losses, epoch_train_accs = [], []
            epoch_soft_losses = []

            # Initialize progress bar
            progress_bar = tqdm(train_loader, ncols=100) if config["logging"]["show_progress"] else train_loader
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
                        soft_loss += distillation_output_loss(new_prev_pred, old_pred, config['training']['temperature']).mean()

                # Total loss
                soft_loss *= config['training']['stability']
                total_loss = hard_loss + soft_loss
                
                # Backward pass and optimization
                total_loss.backward()
                opt.step()

                # Calculate accuracy
                accuracy_batch = get_batch_acc(pred, y)

                # Log metrics to wandb
                wandb.log({
                    'hard_loss': hard_loss.item(), 
                    'soft_loss': soft_loss.item(), 
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

                if config["logging"]["show_progress"]:
                    # Update progress bar description
                    progress_bar.set_description((f'E{e} batch loss:{hard_loss:.2f}, batch acc:{accuracy_batch:>5.1%}'))


            # Evaluate after each epoch on the current task's validation set
            avg_val_loss, avg_val_acc, time = evaluate_model_timed(multitask_model=model,  # model to evaluate
                                    val_loader=val_loader, # task-specific data to evaluate on
                                    prototypes=prototypes ,  # prototypes for the current task
                                    loss_fn=loss_fn,  # loss function
                                    device=device)  # device to run on

            # Log validation metrics to wandb
            wandb.log({
                'val_loss': avg_val_loss, 
                'val_accuracy': avg_val_acc, 
                'epoch': e, 
                'task_id': t,
                'time': time
            })

            # Update metrics
            metrics['epoch_steps'].append(metrics['steps_trained'])
            metrics['train_losses'].extend(epoch_train_losses)
            metrics['train_accs'].extend(epoch_train_accs)
            metrics['val_losses'].append(avg_val_loss)
            metrics['val_accs'].append(avg_val_acc)
            metrics['soft_losses'].extend(epoch_soft_losses)

            if config["logging"]["verbose"]:
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
        if config["logging"]["plot_training"] and len(metrics['val_losses']) > 0:
            training_plot(metrics, show_timesteps=True, results_dir=f'{results_dir}/training-t{t}.png')

        if config["logging"]["verbose"]:
            logger.log(f"Best validation accuracy: {metrics['best_val_acc']:.4f}")
            logger.log(f"Epoch {e} completed in {time:.2f}s")   
        metrics['best_val_acc'] = 0.0

        # Evaluate on all tasks up to current
        metrics_test = test_evaluate_metrics(
                            multitask_model=model,
                            selected_test_sets=data['task_test_sets'][:t+1],
                            task_test_sets=data['task_test_sets'],
                            model_name=f'LwF at t={t}',
                            prev_accs=prev_test_accs,
                            verbose=True,
                            task_metadata=data['task_metadata'],
                            device=device,
                            task_prototypes=data['task_prototypes']
                            )

        # Log test accuracy to wandb
        wandb.log({**metrics_test, 'task_id': t})

        # Append test accuracies
        prev_test_accs.append(metrics_test['task_test_accs'])

        #store the current model as the previous model
        previous_model = model.deepcopy()

    #Log final metrics
    logger.log(f"Task {t} completed!")
    logger.log(f'final metrics: {metrics_test}')
    wandb.summary.update(metrics_test)