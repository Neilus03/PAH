# set up the environment and install any missing packages:
#!pip install torch torchvision numpy scipy matplotlib pandas pillow tqdm MLclf

# PyTorch for building and training neural networks
import torch
from torch import nn, utils
import torch.nn.functional as F

# DataLoader for creating training and validation dataloaders
from torch.utils.data import DataLoader

from collections import OrderedDict

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
from utils import (config_load, seed_everything, setup_dataset_prototype, 
                   evaluate_model_2d, get_batch_acc, test_evaluate_2d, training_plot,
                   distillation_output_loss, TotalVariationLoss, logger)

# Import the HyperCMTL_seq model architecture
from networks.hypernetwork import HyperCMTL_seq_simple_2d
from networks.backbones import ResNet50, MobileNetV2, EfficientNetB0

# Import the wandb library for logging metrics and visualizations
import wandb

### Learning without Forgetting:
from copy import deepcopy # Deepcopy for copying models

# time and logging for logging training progress
import time
import logging
import sys

# Import Optuna for hyperparameter optimization
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

###########################################
# Load config
config = config_load(sys.argv[1])["config"]

device = torch.device(config["misc"]["device"] if torch.cuda.is_available() else "cpu")
seed_everything(config['misc']['seed'])

###########################################
# Define the objective function for Optuna

def objective(trial):
    # Sample hyperparameters from Optuna
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)  # Learning rate
    seed = trial.suggest_int('seed', 0, 1000)  # Random seed
    temperature = trial.suggest_float('temperature', 2.0, 7.0)  # distillation temperature
    stability = trial.suggest_float('stability', 0.5, 5.0)  # stability parameter for soft loss
    task_head_projection_size = trial.suggest_int('task_head_projection_size', 128, 1024, step=128)
    hyper_hidden_features = trial.suggest_int('hyper_hidden_features', 128, 512, step=64)
    hyper_hidden_layers = trial.suggest_int('hyper_hidden_layers', 2, 12, step=1)

    # Override config parameters with trial suggestions
    config['training']['temperature'] = temperature
    config['training']['stability'] = stability
    config['model']['task_head_projection_size'] = task_head_projection_size
    config['model']['hyper_hidden_features'] = hyper_hidden_features
    config['model']['hyper_hidden_layers'] = hyper_hidden_layers
    # Learning rate is applied when creating the optimizer

    num = time.strftime("%Y%m%d-%H%M%S")
    name_run = f"{config['logging']['name']}-{num}-trial{trial.number}"
    results_dir = os.path.join(config["logging"]["results_dir"], name_run)
    os.makedirs(results_dir, exist_ok=True)

    logger_instance = logger(results_dir)

    # Log initial information
    logger_instance.log(f"Starting trial {trial.number} for {config['logging']['name']}")
    logger_instance.log(f"Configuration: {config}")
    logger_instance.log(f"Device: {device}")
    logger_instance.log(f"Random seed: {config['misc']['seed']}")
    logger_instance.log(f"Trial hyperparameters: lr={lr}, temperature={temperature}, stability={stability}, "
                        f"task_head_projection_size={task_head_projection_size}, "
                        f"hyper_hidden_features={hyper_hidden_features}, "
                        f"hyper_hidden_layers={hyper_hidden_layers}")

    data = setup_dataset_prototype(
        dataset_name=config['dataset']['dataset'], 
        data_dir=config["dataset"]["data_dir"], 
        num_tasks=config['dataset']['NUM_TASKS'], 
        val_frac=config['dataset']['VAL_FRAC'], 
        test_frac=config['dataset']['TEST_FRAC'], 
        batch_size=config['dataset']['BATCH_SIZE']
    )

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    num_tasks = len(data['task_metadata'])
    num_classes_per_task = len(data['task_metadata'][0])

    logger_instance.log(f"Using backbone: {config['model']['backbone']}")

    # Update model config dynamically with new hyperparams
    model = HyperCMTL_seq_simple_2d(
        num_tasks=num_tasks,
        num_classes_per_task=num_classes_per_task,
        model_config=config['model'],
        device=device
    ).to(device)

    logger_instance.log(f"Model created!")
    logger_instance.log(f"Model initialized with freeze_backbone={config['model']['frozen_backbone']}, config={config['model']}")

    # If prototypes initialization is enabled
    if config['model']['initialize_prot_w_images']:
        prototypes_inititalization = torch.zeros((num_tasks, num_classes_per_task*config['model']['prototypes_channels']*config['model']['prototypes_size']*config['model']['prototypes_size']))
        fig, ax = plt.subplots(num_tasks, num_classes_per_task, figsize=(num_classes_per_task*3, (num_tasks)*3))
        for t in range(num_tasks):
            if config['model']['prototypes_size'] != 20:
                logger_instance.log(f"Warning: Prototype size is not 20, but {config['model']['prototypes_size']}. Check whether this initialization is correct.")
            prototypes = torch.mean(data['task_prototypes'][t][:, :, 6:26, 6:26], dim=1, keepdim=True)
            prototypes_inititalization[t] = prototypes.reshape(-1)

            for i in range(len(data['task_metadata'][t])):
                ax[t][i].imshow(prototypes[i].permute(1, 2, 0), cmap='gray')
                ax[t][i].axis('off')
                ax[t][i].set_title(data['task_metadata'][t][i])

        plt.savefig(results_dir + '/prototypes.png')
        plt.close()
        model.initialize_embeddings(prototypes_inititalization.to(device))

    # Initialize the previous model
    previous_model = None

    # Initialize optimizer and loss function:
    loss_fn = nn.CrossEntropyLoss()
    tv_loss_fn = TotalVariationLoss()
    opt = torch.optim.AdamW(model.get_optimizer_list(), lr=lr)

    # track metrics for plotting training curves:
    metrics = { 'train_losses': [],
                'train_accs': [],
                'val_losses': [],
                'val_accs': [],
                'epoch_steps': [], # used for plotting val loss at the correct x-position
                'CL_timesteps': [], # used to draw where each new timestep begins
                'best_val_acc': 0.0,
                'steps_trained': 0,
                'soft_losses': [] # distillation loss
              }

    prev_test_accs = []
    prev_test_accs_prot = []

    logger_instance.log(f"Starting training for {config['logging']['name']}")

    with wandb.init(project='HyperCMTL', name=f'{name_run}', config=config) as run:
        if config['model']['initialize_prot_w_images']:
            wandb.log({"all prototypes": wandb.Image(results_dir + '/prototypes.png')})

        # Outer loop over each task, in sequence
        for t, (task_train, task_val) in data['timestep_tasks'].items():
            task_train.num_classes = len(data['timestep_task_classes'][t])
            logger_instance.log(f"Task {t}: {task_train.num_classes} classes\n: {data['task_metadata'][t]}")
            
            # Build train and validation loaders for the current task:
            train_loader, val_loader = [utils.data.DataLoader(dset,
                                            batch_size=config['dataset']['BATCH_SIZE'],
                                            shuffle=True)
                                            for dset in (task_train, task_val)]

            # Evaluate once before training this task
            eval_data = evaluate_model_2d(model, val_loader, loss_fn=loss_fn, device=device, task_metadata=data['task_metadata'], task_id=t, wandb_run=run.id)
            avg_val_loss, avg_val_acc, avg_val_loss_prot, avg_val_acc_prot, time_eval = eval_data

            # Inner loop over the current task:
            for e in range(config['training']['epochs_per_timestep']):
                epoch_train_losses, epoch_train_accs = [], []
                epoch_soft_losses = []

                progress_bar = tqdm(train_loader, ncols=100) if config["logging"]["show_progress"] else train_loader
                
                for batch_idx, batch in enumerate(progress_bar):
                    # Get data from batch
                    x, y, task_ids = batch
                    x, y = x.to(device), y.to(device)
                    task_id = task_ids[0]

                    # zero the gradients
                    opt.zero_grad()

                    # get the predictions from the model
                    pred, pred_prototypes = model(x, task_id)
                    prototypes = model.get_prototypes()
                    y_prototypes = torch.arange(len(data['task_metadata'][int(task_id)]), device=device, dtype=torch.int64)
                    
                    hard_loss = loss_fn(pred, y)
                    prototypes_loss = loss_fn(pred_prototypes, y_prototypes) * config['training']['weight_hard_loss_prototypes']
                    
                    smoothness_loss = tv_loss_fn(prototypes) * config['training']['weight_smoothness_loss']

                    #if previous model exists, calculate distillation loss
                    soft_loss = torch.tensor(0.0, device=device)
                    if previous_model is not None:
                        for old_task_id in range(t):
                            with torch.no_grad():
                                old_pred, old_pred_prot = previous_model(x, old_task_id)
                            new_prev_pred, new_prev_pred_prot = model(x, old_task_id)
                            soft_loss += distillation_output_loss(new_prev_pred, old_pred, config['training']['temperature']).mean().to(device)
                            soft_loss += distillation_output_loss(new_prev_pred_prot, old_pred_prot, config['training']['temperature']).mean().to(device) * config['training']['weight_soft_loss_prototypes']

                    soft_loss *= config['training']['stability']
                    total_loss = hard_loss + soft_loss + prototypes_loss + smoothness_loss
                    
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                    accuracy_batch = get_batch_acc(pred, y)
                    
                    wandb.log({'hard_loss': hard_loss.item(), 
                               'soft_loss': soft_loss.item(), 
                               'train_loss': total_loss.item(), 
                               'prototype_loss': prototypes_loss.item(),
                               'smoothness_loss': smoothness_loss.item(),
                               'epoch': e, 'task_id': t, 'batch_idx': batch_idx, 'train_accuracy': accuracy_batch})

                    # track loss and accuracy:
                    epoch_train_losses.append(hard_loss.item())
                    epoch_train_accs.append(accuracy_batch)
                    epoch_soft_losses.append(soft_loss.item())
                    metrics['steps_trained'] += 1

                    if config["logging"]["show_progress"]:
                        progress_bar.set_description((f'E{e} batch loss:{hard_loss:.2f}, batch acc:{accuracy_batch:>5.1%}'))

                # evaluate after each epoch on the current task's validation set:
                eval_data = evaluate_model_2d(model, val_loader, loss_fn=loss_fn, device=device, task_metadata=data['task_metadata'], task_id=t, wandb_run=run.id)
                avg_val_loss, avg_val_acc, avg_val_loss_prot, avg_val_acc_prot, time_eval = eval_data

                wandb.log({'val_loss': avg_val_loss, 'val_accuracy': avg_val_acc, 
                           'epoch': e, 'task_id': t, 'val_prot_loss': avg_val_loss_prot, 
                           'val_prot_accuracy': avg_val_acc_prot, 'time': time_eval})

                # update metrics:
                metrics['epoch_steps'].append(metrics['steps_trained'])
                metrics['train_losses'].extend(epoch_train_losses)
                metrics['train_accs'].extend(epoch_train_accs)
                metrics['val_losses'].append(avg_val_loss)
                metrics['val_accs'].append(avg_val_acc)
                metrics['soft_losses'].extend(epoch_soft_losses)

                if config["logging"]["show_progress"]:
                    # log end-of-epoch stats:
                    logger_instance.log((f'E{e} loss:{np.mean(epoch_train_losses):.2f}|v:{avg_val_loss:.2f}' +
                                         f'| acc t:{np.mean(epoch_train_accs):>5.1%}|v:{avg_val_acc:>5.1%}'))

                if avg_val_acc > metrics['best_val_acc']:
                    metrics['best_val_acc'] = avg_val_acc

            # mark the end of training on this task for plotting
            metrics['CL_timesteps'].append(metrics['steps_trained'])

            # plot training curves only if validation losses exist
            if config["logging"]["plot_training"] and len(metrics['val_losses']) > 0:
                training_plot(metrics, show_timesteps=True, results_dir=results_dir + f'/training-t{t}.png')

            if config["logging"]["verbose"]:
                logger_instance.log(f"Best validation accuracy: {metrics['best_val_acc']:.4f}")
                logger_instance.log(f"Epoch {e} completed in {time_eval:.2f}s")
            metrics['best_val_acc'] = 0.0
            
            # evaluate on all tasks:
            metrics_test = test_evaluate_2d(
                            multitask_model=model, 
                            selected_test_sets=data['task_test_sets'][:t+1],  
                            task_test_sets=data['task_test_sets'], 
                            prev_accs=prev_test_accs,
                            prev_accs_prot=prev_test_accs_prot,
                            show_taskwise_accuracy=True, 
                            baseline_taskwise_accs=None, 
                            model_name='HyperCMTL_seq + LwF', 
                            verbose=True, 
                            batch_size=config['dataset']['BATCH_SIZE'],
                            results_dir=results_dir,
                            task_id=t,
                            task_metadata=data['task_metadata'],
                            wandb_run=run.id,
                            device=device
                            )
            
            wandb.log({**metrics_test, 'task_id': t})
            
            prev_test_accs.append(metrics_test['task_test_accs'])
            prev_test_accs_prot.append(metrics_test['task_test_accs_prot'])

            # store the current model as the previous model
            previous_model = model.deepcopy()

        # Log final metrics
        final_avg_test_acc = np.mean(metrics_test['task_test_accs'])
        logger_instance.log(f"Final average test accuracy: {final_avg_test_acc:.4f}")
        wandb.summary.update(metrics_test)
        wandb.summary['final_avg_test_acc'] = final_avg_test_acc

    # Return final average test accuracy for Optuna optimization
    return final_avg_test_acc

###########################################
# Run Optuna study

if __name__ == "__main__":
    # Create the Optuna study
    study = optuna.create_study(direction='maximize')  # Maximize validation (or test) accuracy
    study.optimize(objective, n_trials=30)  # Adjust n_trials as needed

    # Create results directory if doesn't exist:
    if not os.path.exists(config["logging"]["results_dir"]):
        os.makedirs(config["logging"]["results_dir"], exist_ok=True)

    # Save figures for the optuna study
    optuna_results_dir = config["logging"]["results_dir"]
    plot_optimization_history(study).write_image(os.path.join(optuna_results_dir, 'optuna_study.png'))
    plot_param_importances(study).write_image(os.path.join(optuna_results_dir, 'optuna_params.png'))

    # Save the .npy file with the optimal hyperparameters found in the study 
    np.save(os.path.join(optuna_results_dir, 'optuna_hyperparameters.npy'), study.best_trial.params)

    # Print the best hyperparameters
    print(f'Best trial: {study.best_trial.params}')
    print(f'Best validation accuracy: {study.best_value}')
