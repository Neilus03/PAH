# set up the environment and install any missing packages:
#!pip install torch torchvision numpy scipy matplotlib pandas pillow tqdm MLclf

# PyTorch for building and training neural networks
import torch
from torch import nn, utils
import torch.nn.functional as F

# DataLoader for creating training and validation dataloaders
from torch.utils.data import DataLoader, BatchSampler

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
from networks.networks_baseline import *

# Import the wandb library for logging metrics and visualizations
import wandb

### Learning without Forgetting:
from copy import deepcopy # Deepcopy for copying models

# time and logging for logging training progress
import time
import logging
import pdb

import random
from torch.utils.data import Sampler
from networks.backbones import ResNet50, MobileNetV2, EfficientNetB0

config = config_load('./configs/baseline.py')
seed_everything(config['misc_config']['seed'])

device = torch.device(config['misc_config']['device'] if torch.cuda.is_available() else "cpu")

### dataset hyperparameters:
VAL_FRAC = config['dataset_config']['VAL_FRAC']
TEST_FRAC = config['dataset_config']['TEST_FRAC']
BATCH_SIZE = config['dataset_config']['BATCH_SIZE']
dataset = config['dataset_config']['dataset']
NUM_TASKS = config['dataset_config']['NUM_TASKS']

### training hyperparameters:
EPOCHS_PER_TIMESTEP = config['training_config']['epochs_per_timestep']
lr     = config['training_config']['lr'] 
l2_reg = config['training_config']['l2_reg']
temperature = config['training_config']['temperature']
stability = config['training_config']['stability']
backbone_name = config['model_config']['backbone']
freeze_backbone = config['training_config']['freeze_backbone']

### metrics and plotting:
plot_training = True   # show training plots after each timestep
show_progress = True   # show progress bars and end-of-epoch metrics
verbose       = True   # output extra info to console

num = time.strftime("%m%d-%H%M%S")
name_run = num + config['logging_config']['name']
results_dir = os.path.join(config['logging_config']['results_dir'], name_run)
os.makedirs(results_dir, exist_ok=True)

logger = logger(results_dir)

# Log initial information
logger.log('Starting training...')
logger.log(f'Training hyperparameters: EPOCHS_PER_TIMESTEP={EPOCHS_PER_TIMESTEP}, lr={lr}, l2_reg={l2_reg}, temperature={temperature}, stability={stability}')
logger.log(f'Training on device: {device}')

### Define preprocessing transform and load a batch to inspect it:
data = setup_dataset(dataset, data_dir='./data', num_tasks=NUM_TASKS, val_frac=VAL_FRAC, test_frac=TEST_FRAC, batch_size=BATCH_SIZE)

timestep_tasks = data['timestep_tasks']
timestep_task_classes = data['timestep_task_classes']
final_test_loader = data['final_test_loader']
task_metadata = data['task_metadata']
task_test_sets = data['task_test_sets']
images_per_class = data['images_per_class']

backbone_dict = {
    'resnet50': ResNet50,
    'mobilenetv2': MobileNetV2,
    'efficientnetb0': EfficientNetB0
}

backbone = backbone_dict[backbone_name](device=device, pretrained=True)
# backbone.num_features = backbone.fc.in_features

if freeze_backbone:
    for param in backbone.parameters():
        param.requires_grad = False

baseline_lwf_model = MultitaskModel_Baseline(backbone, device=device)


# Log the model architecture and configuration
logger.log(f'Model architecture: {baseline_lwf_model}')

logger.log(f"Model initialized with backbone_config={backbone_name}, freeze_backbone={freeze_backbone}")

# Initialize the previous model
previous_model = None

# Initialize optimizer and loss function:
opt = setup_optimizer(baseline_lwf_model, lr=lr, l2_reg=l2_reg, optimizer=config['training_config']['optimizer'])
loss_fn = nn.CrossEntropyLoss()

# track metrics for plotting training curves:
metrics = { 'train_losses': [],
              'train_accs': [],
              'val_losses': [],
                'val_accs': [],
             'epoch_steps': [], # used for plotting val loss at the correct x-position
            'CL_timesteps': [], # used to draw where each new timestep begins
            'best_val_acc': 0.0,
           'steps_trained': 0,
             'soft_losses': [], # distillation loss
          }

prev_test_accs = []

print("Starting training")

def count_optimizer_parameters(optimizer: torch.optim.Optimizer) -> None:
    """
    Prints the number of parameters in each parameter group of the optimizer
    and the total number of unique parameters being optimized.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer instance to inspect.
    """
    total_params = 0
    unique_params = set()
    
    print("\n=== Optimizer Parameter Groups ===")
    for idx, param_group in enumerate(optimizer.param_groups):
        num_params = len(param_group['params'])
        # Calculate the total number of parameters in this group
        num_params_in_group = sum(p.numel() for p in param_group['params'])
        print(f"Parameter Group {idx + 1}: {num_params} parameters, Total Parameters: {num_params_in_group}")
        total_params += num_params_in_group
        # Add to the set of unique parameters to avoid double-counting
        for p in param_group['params']:
            unique_params.add(p)
    
    print(f"Total Optimized Parameters: {total_params} ({sum(p.numel() for p in unique_params)} unique)")
    print("===================================\n")


with wandb.init(project='HyperCMTL', name=f'{name_run}', config=config, group=config['logging_config']['name']) as run:

    for t, (task_train, task_val) in timestep_tasks.items():
        task_train.num_classes = len(timestep_task_classes[t])
        # print(f'task_train.num_classes: {task_train.num_classes}')
        logger.log(f"Training on task id: {t}  (classification between: {task_metadata[t]})")
        if t not in baseline_lwf_model.task_heads:
            task_head = TaskHead_Baseline(input_size=baseline_lwf_model.backbone.num_features,
                                          projection_size=config['model_config']['task_head_projection_size'],
                                          num_classes=task_train.num_classes, 
                                          device=device)
            baseline_lwf_model.add_task(t, task_head)
            opt.add_param_group({'params': task_head.parameters()})
            # count_optimizer_parameters(opt)

        # build train and validation loaders for the current task:
        train_loader, val_loader = [utils.data.DataLoader(data,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True)
                                        for data in (task_train, task_val)]
    

        # inner loop over the current task:
        for e in range(EPOCHS_PER_TIMESTEP):
            epoch_train_losses, epoch_train_accs = [], []
            epoch_soft_losses = []
            
            progress_bar = tqdm(train_loader, ncols=100) if show_progress else train_loader
            num_batches = len(train_loader)
            for batch_idx, batch in enumerate(progress_bar):
                #Get data from batch
                x, y, task_ids = batch
                x, y = x.to(device), y.to(device)
                task_id = task_ids[0]
                
                # zero the gradients
                opt.zero_grad()

                # get the predictions from the model
                pred = baseline_lwf_model(x, task_id)
                hard_loss = loss_fn(pred, y)
                
                #if previous model exists, calculate distillation loss
                soft_loss = 0.0
                if previous_model is not None:
                    for old_task_id in range(t):
                        with torch.no_grad():
                            old_pred = previous_model(x, old_task_id)
                        new_prev_pred = baseline_lwf_model(x, old_task_id)
                        soft_loss += distillation_output_loss(new_prev_pred, old_pred, temperature).mean()
                
                #add the distillation loss to the total loss
                total_loss = hard_loss + stability * soft_loss
                
                wandb.log({'hard_loss': hard_loss.item(), 'soft_loss': float(soft_loss), 'train_loss': total_loss.item(), 'epoch': e, 'task_id': t, 'batch_idx': batch_idx})

                #backpropagate the loss
                total_loss.backward()
                opt.step()
                
                # track loss and accuracy:
                epoch_train_losses.append(hard_loss.item())
                epoch_train_accs.append(get_batch_acc(pred, y))
                epoch_soft_losses.append(soft_loss.item() if isinstance(soft_loss, torch.Tensor) else soft_loss)
                metrics['steps_trained'] += 1
                
                if show_progress:
                    # show loss/acc of this batch in progress bar:
                    progress_bar.set_description((f'E{e} batch loss:{hard_loss:.2f}, batch acc:{epoch_train_accs[-1]:>5.1%}'))

            # evaluate after each epoch on the current task's validation set:
            avg_val_loss, avg_val_acc, time = evaluate_model_timed(baseline_lwf_model, val_loader, loss_fn, device=device)
            
            wandb.log({'val_loss': avg_val_loss, 'val_accuracy': avg_val_acc, 'epoch': e, 'task_id': t, 'time': time})

            ### update metrics:
            metrics['epoch_steps'].append(metrics['steps_trained'])
            metrics['train_losses'].extend(epoch_train_losses)
            metrics['train_accs'].extend(epoch_train_accs)
            metrics['val_losses'].append(avg_val_loss)
            metrics['val_accs'].append(avg_val_acc)
            metrics['soft_losses'].extend(epoch_soft_losses)
            
            if show_progress:
                # print end-of-epoch stats:
                print((f'E{e} loss:{np.mean(epoch_train_losses):.2f}|v:{avg_val_loss:.2f}' +
                                    f'| acc t:{np.mean(epoch_train_accs):>5.1%}|v:{avg_val_acc:>5.1%} in {time:.2f}s'))

            if avg_val_acc > metrics['best_val_acc']:
                metrics['best_val_acc'] = avg_val_acc
            
        # this one is important for nice plots:
        metrics['CL_timesteps'].append(metrics['steps_trained']) 

        # plot training curves only if validation losses exist
        if plot_training and len(metrics['val_losses']) > 0:
            training_plot(metrics, show_timesteps=True)

        if verbose:
            print(f'Best validation accuracy: {metrics["best_val_acc"]:.2%}\n')
        metrics['best_val_acc'] = 0.0   

        # evaluate on all tasks:
        metrics_test = test_evaluate_metrics(baseline_lwf_model, task_test_sets[:t+1],
                                  task_test_sets=task_test_sets, 
                                model_name=f'LwF at t={t}', 
                                prev_accs = prev_test_accs,
                                #baseline_taskwise_accs = baseline_taskwise_test_accs, 
                                verbose=True,
                                task_metadata=task_metadata,
                                device=device)
        
        wandb.log({**metrics_test, 'task_id': t})

        prev_test_accs.append(metrics_test['task_test_accs'])
        
        #store the current baseline_lwf_model as the previous baseline_lwf_model
        previous_model = deepcopy(baseline_lwf_model)

    print(f'Final metrics: {metrics_test}')
    wandb.summary.update(metrics_test)
   